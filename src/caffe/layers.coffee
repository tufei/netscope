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

layers.PriorBox =
class @PriorBoxLayer
    constructor: (attribs) ->
        params = attribs?.prior_box_param
        unless params?.min_size?
            throw 'PriorBox layer must have min_size'
        @flip = getValueOrDefault params?.flip, false
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
        unless params?.shape?
            throw 'Reshape layer requires shape parameter'
        @shape = utils.asArray params.shape
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
        console.log "#{partial_size} #{@shape}"
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

layers.Accuracy =
class @AccuracyLayer
    constructor: (attribs) ->
        params = attribs?.accuracy_param?
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

layers.Data =
class @DataLayer
    constructor: (attribs) ->
        @defaultBatchSize = 1
        @defaultChannels  = 3
        @outputShape = @tryExtractShapes attribs

    inferShapes: (bottoms, tops) =>
        unless tops?[0]? then return
        @checkParameters bottoms, tops
        tops[0].shape = @outputShape[..]
        tops[1].shape = @outputShape[..0] if tops[1]

    checkParameters: (bottoms, tops) =>
        unless @outputShape?
            throw "Can't extract data shape from Data layer"
        if bottoms?.length > 0
            throw "Data layer doesn't expect any input."
        unless tops?.length in [1, 2]
            throw 'Outputs number of Data layer must be equal to one or two.'

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
    (/mvn/i.test       lt) or
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
