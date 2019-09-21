utils = require '../utils/utils.coffee'

areShapesEqual = (x, y) ->
    if x.length != y.length
        return false
    for i in [0...x.length]
        if x[i] != y[i]
            return false
    return true

getValueOrDefault = (param, defaultValue) ->
    if param? then param else defaultValue

extractKernelSizes =(params) ->
    params.kernel_size or [ params.kernel_h, params.kernel_w ]

extractPaddingSizes = (params) ->
    if params.pad?
        return params.pad
    if (not params.pad_h?) and (not params.pad_w?)
        return 0
    return [
        getValueOrDefault params.pad_h, 0
        getValueOrDefault params.pad_w, 0
    ]

extractStrideSizes = (params) ->
    if params.stride?
        return params.stride
    if (not params.stride_h?) and (not params.stride_w?)
        return 1
    return [
        getValueOrDefault params.stride_h, 1
        getValueOrDefault params.stride_w, 1
    ]

getParameterAsArray = (parameter, requiredLength, name) ->
    if utils.typeIsArray parameter
        if parameter.length != requiredLength
            throw "Dimensions of the '#{name}' parameter " +
                  "must be equal to #{requiredLength}."
        return parameter
    return (parameter for i in [0...requiredLength])

shapesToString = (inputShapes) ->
    text = '['
    for shape in inputShapes
        text += " [ #{shape} ]"
    text += ' ]'
    return text


layers = {}
layers.Uniform =
class @UniformLayer
    inferShapes: (bottoms, tops) ->
        unless tops?[0]? then return
        # Assume 'Uniform' layer doen't change the output shape
        # We interpret all currently unsupported layer as 'Uniform'
        for i in [0...tops.length]
            tops[i].shape = bottoms[i].shape[..]

layers.Loss =
class @LossLayer
    inferShapes: (bottoms, tops) ->
        unless tops?[0]? then return
        # Loss layer always returns scalar
        tops[0].shape = [ 1 ]

layers.ROIAlign =
class @ROIAlignLayer
    constructor: (attribs) ->
        params = attribs?.roi_align_param
        unless params?.pooled_h? && params?.pooled_w?
            throw 'ROIAlign layer requires pooled_h/w parameter.'
        @pooled_h = params.pooled_h
        @pooled_w = params.pooled_w

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        tops[0].shape = [ ]
        tops[0].shape.push(bottoms[1].shape[0], bottoms[0].shape[1], @pooled_h, @pooled_w)

    checkParameters: (bottoms, tops) =>
        unless bottoms?.length == 2
            throw "ROIAlign layer must have two inputs."
        unless tops?.length == 1
            throw 'Outputs number of Flatten layer must be equal to one.'
        unless @pooled_h > 0 && @pooled_w > 0
            throw "Pooled height #{@pooled_h} and/or width #{@pooled_w} of ROIAlign layer invalid."

layers.Flatten =
class @FlattenLayer
    constructor: (attribs) ->
        params = attribs?.flatten_param
        @axis = getValueOrDefault params?.axis, 1
        @end_axis = getValueOrDefault params?.end_axis, -1

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        @axis = bottoms[0].shape.length + @axis if @axis < 0
        @end_axis = bottoms[0].shape.length + @end_axis if @end_axis < 0
        tops[0].shape = [ ]
        for i in [0...@axis]
            tops[0].shape.push(bottoms[0].shape[i])
        size = 1
        for i in [@axis..@end_axis]
            size = size * bottoms[0].shape[i]
        tops[0].shape.push(size)
        for i in [@end_axis + 1...bottoms[0].shape.length]
            tops[0].shape.push(bottoms[0].shape[i])

    checkParameters: (bottoms, tops) =>
        unless bottoms?.length == 1
            throw "Flatten layer must have one input."
        unless tops?.length == 1
            throw 'Outputs number of Flatten layer must be equal to one.'
        unless @axis < bottoms[0].shape.length && @end_axis < bottoms[0].shape.length
            throw "Axis #{@axis} and/or End-Axis #{@end_axis} of Flatten layer larger than #{bottoms[0].shape.length}."

layers.Interp =
class @InterpLayer
    constructor: (attribs) ->
        params = attribs?.interp_param
        @pad_begin = getValueOrDefault params?.pad_begin, 0
        @pad_end = getValueOrDefault params?.pad_end, 0
        if params?.shrink_factor? || params.zoom_factor?
            if params?.height? || params?.width?
                throw 'cannot define zoom/shrink factor and width/height at the same time.'
            if params?.shrink_factor?
                @shrink_factor = params.shrink_factor
            else
                @shrink_factor = 0

            if params?.zoom_factor?
                @zoom_factor = params.zoom_factor
            else
                @zoom_factor = 0
        else
            if params?.height? && params?.width?
                @interp_w = params.width
                @interp_h = params.height
                @zoom_factor = 0
                @shrink_factor = 0
            else
                throw 'has to define width and height at the same time.'

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        if @zoom_factor != 0 || @shrink_factor != 0
            @interp_h = bottoms[0].shape[2] + @pad_begin + @pad_end
            @interp_w = bottoms[0].shape[3] + @pad_begin + @pad_end
            if @shrink_factor != 0
              @interp_h = (@interp_h - 1) / @shrink_factor + 1 
              @interp_w = (@interp_w - 1) / @shrink_factor + 1 
            if @zoom_factor !=  0
              @interp_h = @interp_h + (@interp_h - 1) * (@zoom_factor - 1) 
              @interp_w = @interp_w + (@interp_w - 1) * (@zoom_factor - 1) 
        tops[0].shape = [ ]
        tops[0].shape.push(bottoms[0].shape[0], bottoms[0].shape[1], @interp_h, @interp_w)

    checkParameters: (bottoms, tops) =>
        unless bottoms?.length == 1
            throw "Interp layer must have one input."
        unless tops?.length == 1
            throw 'Outputs number of Interp layer must be equal to one.'
        unless @pad_begin <= 0 && @pad_end <= 0
            throw "pad_begin #{@pad_begin} and pad_end #{@pad_end} has to be non-positive for now."
        unless @zoom_factor == 0 || @zoom_factor >= 1
            throw "zoom_factor #{@zoom_factor} must be not less than 1."
        unless @shrink_factor == 0 || @shrink_factor >= 1
            throw "shrink_factor #{@shrink_factor} must be not less than 1."

layers.Upsample =
class @UpsampleLayer
    constructor: (attribs) ->
        params = attribs?.upsample_param
        @pad_h = getValueOrDefault params?.pad_out_h, 'false'
        @pad_w = getValueOrDefault params?.pad_out_w, 'false'
        if params?.upsample_h? && params?.upsample_w?
            @upsample_h = params.upsample_h
            @upsample_w = params.upsample_w
            if params?.scale?
                @scale_h = params.scale
                @scale_w = params.scale
        else
            @upsample_h = -1
            @upsample_w = -1
            if params?.scale_h? && params?.scale_w?
                if params?.scale?
                    throw 'cannot define scale and scale_h/w at the same time.'
                @scale_h = params.scale_h
                @scale_w = params.scale_w
            else
                unless params?.scale?
                    throw 'Upsample layer needs either scale or upsampled resolution.'
                @scale_h = params.scale
                @scale_w = params.scale

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        pad_out_h = if @pad_h == 'false' then 0 else 1
        pad_out_w = if @pad_w == 'false' then 0 else 1
        if (pad_out_h == 1 && @scale_h != 2) || (pad_out_w == 1 && @scale_w != 2)
            throw 'Padding compensation requires scale ratio to be 2.'
        if @upsample_h == -1 && @upsample_w == -1
            @upsample_h = bottoms[0].shape[2] * @scale_h - pad_out_h
            @upsample_w = bottoms[0].shape[3] * @scale_w - pad_out_w
        tops[0].shape = [ ]
        tops[0].shape.push(bottoms[0].shape[0], bottoms[0].shape[1], @upsample_h, @upsample_w)

    checkParameters: (bottoms, tops) =>
        unless bottoms?.length == 2
            throw 'Inputs number of Upsample layer must be equal to two.'
        unless bottoms[0]?.shape?.length == 4 && bottoms[1]?.shape?.length == 4
            throw 'Upsample layer bottoms must have dimension of 4.'
        unless tops?.length == 1
            throw 'Outputs number of Upsample layer must be equal to one.'
        for i in [0...bottoms[0].shape]
            unless bottoms[0].shape[i] == bottoms[1].shape[i]
                throw "Dimension #{i} of Upsample layer: #{bottoms[0].shape[i]} != #{bottoms[1]/shape[i]}."

layers.PriorBox =
class @PriorBoxLayer
    constructor: (attribs) ->
        params = attribs?.prior_box_param
        unless params?.min_size?
            throw 'PriorBox layer must have min_size'
        @flip = getValueOrDefault params?.flip, 'false'
        min_size = utils.asArray params.min_size
        @numMinSizes = min_size.length
        if params.max_size?
            max_size = utils.asArray params.max_size
            @numMaxSizes = max_size.length
        else
            @numMaxSizes = 0
        if params.aspect_ratio?
            aspect_ratio = utils.asArray params.aspect_ratio
            @numAspectRatios = aspect_ratio.length
        else
            @numAspectRatios = 0
        if @flip == 'true'
            @numAspectRatios *= 2
        @numAspectRatios += 1

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        tops[0].shape = [ ]
        tops[0].shape.push(1, 2)
        num_priors = @numMinSizes * @numAspectRatios + @numMaxSizes
        tops[0].shape.push(bottoms[0].shape[2] * bottoms[0].shape[3] * 4 * num_priors)

    checkParameters: (bottoms, tops) =>
        unless bottoms[0]?.shape?.length == 4
            throw 'PriorBox layer bottom must have dimension of 4.'
        unless tops?.length == 1
            throw 'Outputs number of PriorBox layer must be equal to one.'

layers.Reshape =
class @ReshapeLayer
    constructor: (attribs) ->
        params = attribs?.reshape_param
        unless params?.shape?.dim?
            throw 'Reshape layer requires shape dimension parameter'
        @shape = utils.asArray params.shape.dim
        @axis = getValueOrDefault params.axis, 0
        @num_axes = getValueOrDefault params.num_axes, -1

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        total_axes = bottoms[0].shape.length
        if @axis < 0
            @axis = total_axes + 1 + @axis
        if @num_axes < 0
            end_axis = total_axes + @num_axes
        else
            end_axis = @axis + @num_axes
        total_size = 1
        for i in [@axis..end_axis]
            total_size *= bottoms[0].shape[i]
        tops[0].shape = [ ]
        for i in [0...@axis]
            tops[0].shape.push(bottoms[0].shape[i])
        partial_size = 1
        for i in [0...@shape.length]
            if @shape[i] == 0
                tops[0].shape.push(bottoms[0].shape[@axis + i])
                partial_size *= bottoms[0].shape[@axis + i]
            else
                if @shape[i] == -1
                    tops[0].shape.push(-1)
                else
                    tops[0].shape.push(@shape[i])
                    partial_size *= @shape[i]
        infer_size = total_size // partial_size
        unless infer_size * partial_size == total_size
            throw "#{infer_size} * #{partial_size} != #{total_size}"
        index = tops[0].shape.indexOf(-1)
        tops[0].shape.splice(index, 1, infer_size)
        for i in [end_axis + 1...bottoms[0].shape.length]
            tops[0].shape.push(bottoms[0].shape[i])

    checkParameters: (bottoms, tops) =>
        unless bottoms?.length == 1
            throw "Reshape layer must have one input."
        unless tops?.length == 1
            throw 'Outputs number of Reshape layer must be equal to one.'
        num_minus1 = 0
        for i in [0...@shape.length]
            if @shape[i] == -1 then num_minus1 += 1
        unless num_minus1 < 2
            throw 'Only one dimension of Reshape layer can be inferred.'

layers.Permute =
class @PermuteLayer
    constructor: (attribs) ->
        params = attribs?.permute_param
        unless params?.order? then return
        @orders = params.order

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        unless @orders?
            @orders = [0...bottoms[0].shape.length]
        unless @orders.length == bottoms[0].shape.length
            original_orders = [0...bottoms[0].shape.length]
            for i in @orders
                index = original_orders.indexOf(i)
                original_orders.splice(index, 1)
            @orders = @orders.concat(original_orders)
        tops[0].shape = [ ]
        for i in [0...bottoms[0].shape.length]
            tops[0].shape.push(bottoms[0].shape[@orders[i]])

    checkParameters: (bottoms, tops) =>
        unless bottoms?.length == 1
            throw "Permute layer must have one input."
        unless tops?.length == 1
            throw 'Outputs number of Permute layer must be equal to one.'
        unless @orders?.length <= bottoms[0].shape.length
            throw "Order rank #{@orders.length} of Permute layer exceeds blob dimension."
        for i in @orders
            unless i < bottoms[0].shape.length
                throw "Axis #{i} of Permute layer larger than #{bottoms[0].shape.length}."

layers.DetectionOutput =
class @DetectionOutputLayer
    constructor: (attribs) ->
        params = attribs?.detection_output_param
        unless params?.num_classes?
            throw 'DetectionOutput layer must have num_classes.'
        @num_classes = params.num_classes
        @keep_top_k = getValueOrDefault params.keep_top_k, 1
        if params.share_location? and params.share_location == 'true'
            @num_loc_classes = 1
        else
            @num_loc_classes = @num_classes

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        tops[0].shape = [ ]
        tops[0].shape.push(@keep_top_k, 7)

    checkParameters: (bottoms, tops) =>
        unless bottoms?.length == 3
            throw 'DetectionOutput layer must have three inputs.'
        unless tops?.length == 1
            throw 'Outputs number of DetectionOutput layer must be equal to one.'
        num_priors = bottoms[2].shape[2] // 4
        unless num_priors * @num_loc_classes * 4 == bottoms[0].shape[1]
            throw "#{num_priors} * #{@num_loc_classes} * 4 != #{bottoms[0].shape[1]}"
        unless num_priors * @num_classes == bottoms[1].shape[1]
            throw "#{num_priors} * #{@num_classes} != #{bottoms[1].shape[1]}"

layers.Embed =
class @EmbedLayer
    constructor: (attribs) ->
        params = attribs?.embed_param
        unless params?.num_output?
            throw 'Embed layer must have num_output.'
        @num_output = params?.num_output
        @input_dim = params?.input_dim

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        tops[0].shape = bottoms[0].shape
        tops[0].shape.push(@num_output)

    checkParameters: (bottoms, tops) =>
        unless bottoms?.length == 1
            throw 'Embed layer must have one input.'
        unless tops?.length == 1
            throw 'Outputs number of Embed layer must be equal to one.'
        unless @num_output > 0 && @input_dim > 0
            throw "#{@num_output} and/or #{@input_dim} of Embed layer invalid."

layers.Accuracy =
class @AccuracyLayer
    constructor: (attribs) ->
        params = attribs?.accuracy_param
        @axis = getValueOrDefault params?.axis, 1

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        tops[0].shape = [ 1 ]
        tops[1].shape = [ bottoms[0].shape[ @axis ] ] if tops[1]

    checkParameters: (bottoms, tops) =>
        unless bottoms?.length == 2
            throw "Accuracy layer must have two inputs."
        unless tops?.length in [1, 2]
            throw 'Outputs number of Accuracy layer must be equal to one or two.'

layers.ArgMax =
class @ArgMaxLayer
    constructor: (attribs) ->
        params = attribs?.argmax_param
        @out_max_val = getValueOrDefault params?.out_max_val, 'false'
        @axis = params?.axis
        @top_k = getValueOrDefault params?.top_k, 1

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        @num_top_axes = if bottoms[0].shape.length < 3 then 3 else bottoms[0].shape.length
        tops[0].shape = [ ]
        for i in [0...@num_top_axes]
            tops[0].shape.push(1)
        if @axis?
            tops[0].shape = bottoms[0].shape[..]
            tops[0].shape[@axis] = @top_k
        else
            tops[0].shape[0] = bottoms[0].shape[0]
            tops[0].shape[2] = @top_k
            if @out_max_val == 'true'
                tops[0].shape[1] = 2

    checkParameters: (bottoms, tops) =>
        unless bottoms?.length == 1
            throw 'ArgMax layer must have one input.'
        unless tops?.length == 1
            throw 'Outputs number of ArgMax layer must be equal to one.'
        if @axis?
            if @axis < 0 then @axis = bottoms[0].shape.length + @axis
            unless @axis <= bottoms[0].shape.length
                throw "Axis #{@axis} of ArgMax layer invalid."
            unless @top_k <= bottoms[0].shape[@axis]
                throw "#{@top_k} is greater than #{bottoms[0].shape[@axis]} in ArgMax layer."
        else
            size = 1
            for i in [0...bottoms[0].shape.length]
                size *= bottoms[0].shape[i]
            unless @top_k <= size
                throw "#{@top_k} is greater than #{size} in ArgMax layer."

layers.Data =
class @DataLayer
    constructor: (attribs) ->
        @defaultBatchSize = 1
        @defaultChannels  = 3
        @outputShape = @tryExtractShapes attribs

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        if tops.length <= 2
            tops[0].shape = @outputShape[..]
            tops[1].shape = @outputShape[..0] if tops[1]
        else
            if tops.length == @outputShape.length
                for i in [0...tops.length]
                    tops[i].shape = @outputShape[i].dim
            else
                for i in [0...tops.length]
                    tops[i].shape = @outputShape[4 * i...4 * (i + 1)]

    checkParameters: (bottoms, tops) =>
        unless @outputShape?
            throw "Can't extract data shape from Data layer"
        if bottoms?.length > 0
            throw "Data layer doesn't expect any input."
        unless tops?.length <= 6
            throw 'Outputs number of Data layer must be no greater than six.'
        if tops?.length > 2
            unless tops?.length * 4 == @outputShape.length || (tops?.length == @outputShape.length && @outputShape[0].dim?)
                throw 'Shapes of Data layer outputs not fully defined.'

    tryExtractShapes: (attribs) =>
        shape = attribs?.input_param?.shape?.dim
        unless shape?
            shape = attribs?.input_param?.shape
        unless shape?
            shape = attribs?.shape
        unless shape?
            shape = @tryExtractShapeFromTransformParam attribs
        unless shape?
            shape = @tryExtractShapeFromMemoryDataLayer attribs
        return shape

    tryExtractShapeFromTransformParam: (attribs) =>
        cropSize = attribs.transform_param?.crop_size
        if cropSize?
            channels = @defaultChannels
            channels = 1 if attribs.transform_param.force_gray
            return [@defaultBatchSize, channels, cropSize, cropSize]

    tryExtractShapeFromMemoryDataLayer: (attribs) =>
        param = attribs?.memory_data_param
        batch_size = param.batch_size or @defaultBatchSize
        channels   = param.channels or @defaultChannels
        height     = param.height
        width      = param.width
        if height? and width?
            return [batch_size, channels, height, width]

class ConvolutionLayerBase
    constructor: (@name, attribs) ->
        params = attribs?.convolution_param
        unless params?
            throw "#{@name} layer must have convolution_param."
        @filters  = params.num_output
        @padding  = extractPaddingSizes params
        @stride   = extractStrideSizes  params
        @kernel   = extractKernelSizes  params
        @dilation = getValueOrDefault params.dilation, 1
        @axis     = getValueOrDefault params.axis, 1

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        # Convolution layer behaviour is alligned with Caffe
        # The layer processes each bottom -> top pair independently
        for i in [0...tops.length]
            @inferShapesForOneBlob bottoms[i], tops[i]

    inferShapesForOneBlob: (bottom, top) =>
        inputShape = bottom.shape
        outputShape = inputShape[..]
        succeedingDimensions = inputShape[@axis + 1..]
        sucDimLength = succeedingDimensions.length
        padding  = getParameterAsArray @padding,  sucDimLength, 'padding'
        kernel   = getParameterAsArray @kernel,   sucDimLength, 'kernel'
        stride   = getParameterAsArray @stride,   sucDimLength, 'stride'
        dilation = getParameterAsArray @dilation, sucDimLength, 'dilation'
        @inferShapesForOneBlobInternal inputShape, outputShape, padding,
                                       kernel, stride, dilation
        top.shape = outputShape

    inferShapesForOneBlobInternal: (input, output, padding, kernel, stride, dilation) =>
        # Assume 'input' and 'output' are shapes
        undefined

    checkParameters: (bottoms, tops) =>
        unless @filters?
            throw "#{@name} layer must have num_output parameter."
        if not @kernel? and (not @kernel[0]? or not @kernel[1]?)
            console.log @kernel
            throw "#{@name} kernel sizes must be set."
        unless bottoms?
            throw "#{@name} layer received undefined bottom blobs."
        if bottoms.length != tops.length
            throw "#{@name} layer can process number of top blobs which is equal to " +
                  "the number of bottom blobs, but received #{tops.length} top blobs and " +
                  "#{bottoms.length} bottom blobs."

layers.Convolution =
class @ConvolutionLayer extends ConvolutionLayerBase
    constructor: (attribs) ->
        super 'Convolution', attribs

    inferShapesForOneBlobInternal: (input, output, padding, kernel, stride, dilation) =>
        output[@axis] = @filters
        for i in [@axis + 1...input.length]
            ii = i - @axis - 1
            kernelExtent = dilation[ii] * (kernel[ii] - 1) + 1;
            outDim = (input[i] + 2 * padding[ii] - kernelExtent) / stride[ii] + 1
            output[i] = Math.floor outDim

layers.Deconvolution =
class @DeconvolutionLayer extends ConvolutionLayerBase
    constructor: (attribs) ->
        super 'Deconvolution', attribs

    inferShapesForOneBlobInternal: (input, output, padding, kernel, stride, dilation) =>
        output[@axis] = @filters
        for i in [@axis + 1...input.length]
            ii = i - @axis - 1
            kernelExtent = dilation[ii] * (kernel[ii] - 1) + 1;
            outDim = stride[ii] * (input[i] - 1) + kernelExtent - 2 * padding[ii]
            output[i] = Math.floor outDim

layers.Pooling =
class @PoolingLayer
    constructor: (attribs) ->
        @spatialDimSize = 2
        params = attribs?.pooling_param
        if not params?
            throw 'Pooling layer must have pooling_param.'
        @padding = extractPaddingSizes params
        @stride  = extractStrideSizes  params
        @kernel  = extractKernelSizes  params
        @isGlobalPooling = getValueOrDefault params.global_pooling, false
        # Caffe Pooling layer works only with two last axes, so pool will be
        # applied to dim - 2 and dim - 1 axes.

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        # Caffe pooling implementation works only with the single bottom -> top
        # pair. Blob tops[1] stores the output pooling mask if tops.length > 1.
        @checkParameters bottoms, tops
        inputShape = bottoms[0].shape
        outputShape = inputShape[..]
        padding = getParameterAsArray @padding, @spatialDimSize, 'padding'
        stride  = getParameterAsArray @stride,  @spatialDimSize, 'stride'
        kernel  = @getKernelSizes inputShape
        for i in [0...@spatialDimSize]
            ii = inputShape.length - @spatialDimSize + i
            outDim = (inputShape[ii] + 2 * padding[i] - kernel[i]) / stride[i]
            outDimRounded = (Math.floor(Math.ceil outDim)) + 1
            if (outDimRounded - 1) * stride[i] >= inputShape[ii] + padding[i]
                outDimRounded--
            outputShape[ii] = outDimRounded
        tops[0].shape = outputShape
        tops[1].shape = outputShape[..] if tops[1]

    checkParameters: (bottoms, tops) =>
        if not @kernel? and (not @kernel[0]? or not @kernel[1]?)
            throw 'Pooling layer must have kernel_size parameter.'
        unless bottoms?
            throw 'Pooling layer received undefined bottom blobs.'
        if bottoms.length != 1
            throw "Pooling layer can process exactly one input, " +
                  "but received #{bottoms.length} input shapes."
        unless tops.length in [1, 2]
            throw "Pooling layer produces single output shape or two equal " +
                  "shapes if the second top shape is specified."

    getKernelSizes: (inputShape) =>
        if @isGlobalPooling
            kernel = inputShape[-@spatialDimSize..]
        else
            kernel = getParameterAsArray @kernel, @spatialDimSize, 'kernel'
        return kernel


layers.InnerProduct =
class @InnerProductLayer
    constructor: (attribs) ->
        params = attribs?.inner_product_param
        if not params?
            throw 'InnerProduct layer must have inner_product_param.'
        @numOutput = params.num_output
        @axis = getValueOrDefault params.axis, 1

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        inputShape = bottoms[0].shape
        outputShape = inputShape[...@axis]
        outputShape[@axis] = @numOutput
        tops[0].shape = outputShape

    checkParameters: (bottoms, tops) =>
        if not @numOutput?
            throw 'InnerProduct layer must have num_output parameter.'
        unless bottoms?
            throw 'InnerProduct layer received undefined bottom blobs.'
        if bottoms.length != 1 or tops.length != 1
            throw "InnerProduct layer can accept and produce exactly one blob, but " +
                  "received #{bottoms.length} bottoms blobs and #{tops.length} top blobs."


layers.Concat =
class @ConcatLayer
    constructor: (attribs) ->
        params = attribs?.concat_param
        axis   = params?.concat_dim
        axis  ?= params?.axis
        @axis  = getValueOrDefault axis, 1

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        firstInputShape = bottoms[0].shape
        outputShape = firstInputShape[..]
        outputShape[@axis] = 0
        for bottom in bottoms
            outputShape[@axis] += bottom.shape[@axis]
        tops[0].shape = outputShape

    checkParameters: (bottoms, tops) =>
        unless bottoms?[0]?
            throw 'Concat layer must have at least one bottom blob.'
        firstShape = bottoms[0].shape
        inputShapes = (bottom.shape for bottom in bottoms)
        for shape in inputShapes
            unless @checkInputShapeAxes firstShape, shape
                throw "Concat layer received incorrect input shapes: " +
                      "#{shapesToString(inputShapes)}. " +
                      "All axes except axis along which concatenation " +
                      "is performing must have the same sizes."

    checkInputShapeAxes: (firstShape, shape) =>
        if firstShape.length != shape.length
            return false
        for i in [0...shape.length]
            if i != @axis and firstShape[i] != shape[i]
                return false
        return true

layers.Eltwise =
class @EltwiseLayer
    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        firstInputShape = bottoms[0].shape
        tops[0].shape = firstInputShape[..]

    checkParameters: (bottoms, tops) =>
        unless bottoms?[0]?
            throw 'Eltwise layer must have at least one input.'
        inputShapes = (bottom.shape for bottom in bottoms)
        firstShape = inputShapes[0]
        for shape in inputShapes
            unless areShapesEqual firstShape, shape
                throw "Eltwise layer received incorrect input shapes: " +
                      "#{shapesToString(inputShapes)}. " +
                      "All axes must have the same sizes."

layers.Crop =
class @CropLayer
    constructor: (attribs) ->
        params = attribs.crop_param
        @axis = getValueOrDefault params?.axis, 0

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        outputShape = bottoms[0].shape[..]
        for i in [@axis...outputShape.length]
            outputShape[i] = bottoms[1].shape[i]
        tops[0].shape = outputShape

    checkParameters: (bottoms, tops) =>
        if bottoms?.length != 2
            throw 'Crop layer must have exactly two bottom blobs.'


layers.Split =
class @SplitLayer
    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        outputShape = bottoms[0].shape[..]
        for i in [0...tops.length]
            tops[i].shape = outputShape

    checkParameters: (bottoms, tops) =>
        if bottoms?.length != 1
            throw 'Split layer must have exactly one bottom blob.'


isLossLayer = (layerType) ->
    /loss/i.test layerType

isDataLayer = (layerType) ->
    (/input/i.test layerType) or
    (/data/i.test layerType)

isUniformLayer = (lt) ->
    (/relu/i.test      lt) or
    (/relu6/i.test     lt) or
    (/prelu/i.test     lt) or
    (/elu/i.test       lt) or
    (/sigmoid/i.test   lt) or
    (/tanh/i.test      lt) or
    (/abs/i.test       lt) or
    (/power/i.test     lt) or
    (/exp/i.test       lt) or
    (/log/i.test       lt) or
    (/bnll/i.test      lt) or
    (/threshold/i.test lt) or
    (/bias/i.test      lt) or
    (/scale/i.test     lt) or
    (/lrn/i.test       lt) or
    (/dropout/i.test   lt) or
    (/batchnorm/i.test lt) or
    (/bn/i.test        lt) or
    (/mvn/i.test       lt) or
    (/quantize/i.test  lt) or
    (/softmax/i.test   lt)

getLayerType = (layerTypeName) ->
    if isUniformLayer layerTypeName
        return layers.Uniform
    if isDataLayer layerTypeName
        return layers.Data
    if isLossLayer layerTypeName
        return layers.Loss
    layerType = layers[layerTypeName]
    unless layerType?
        layerTypeNameTitle = utils.toTitleCase layerTypeName
        layerType = layers[layerTypeNameTitle]
    unless layerType?
        throw "Unsupported layer type: '#{layerTypeName}'."
    return layerType

exports.inferTopShapes = (node) ->
    try
        LayerType = getLayerType node.type
        layer = new LayerType node.attribs
        layer.inferShapes node.bottoms, node.tops
        return (top.shape for top in node.tops)
    catch e
        throw "Can't infer output shape of the '#{node.name}' " +
              "layer of type '#{node.type}'. " + e
