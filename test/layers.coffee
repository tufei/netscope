assert = require 'assert'
layers = require '../src/caffe/layers.coffee'
utils  = require '../src/utils/utils.coffee'

runLayerTasks = (tasks, nameFunction, compareFunction) ->
    tasks.forEach (task) ->
        it nameFunction(task), ->
            compareFunction task

shapeToArrayOfShapes = (shapeOrShapes) ->
    if utils.typeIsArray shapeOrShapes[0]
        return shapeOrShapes
    return [ shapeOrShapes ]

compareLayerOutput = (LayerType, parameterMaker, task) ->
    [inputShapeOrShapes, expectingOutputShapeOrShapes, params] = task
    caffeParams = parameterMaker utils.asArray(params)...
    layer = new LayerType caffeParams
    inputShapes = shapeToArrayOfShapes inputShapeOrShapes
    expectingOutputShapes = shapeToArrayOfShapes expectingOutputShapeOrShapes
    bottoms = ( { shape: s } for s in inputShapes )
    tops = ( { shape: null } for s in expectingOutputShapes )
    layer.inferShapes bottoms, tops
    actualOutputShapes = (top.shape for top in tops)
    assert.deepEqual expectingOutputShapes, actualOutputShapes

stringifyConvParams = (filters, kernels, strides, paddings,
                       useKernelHW = false,
                       useStrideHW = false,
                       usePaddingHW = false) ->
    kernelsStr  = utils.asArray(kernels).join 'x'
    stridesStr  = utils.asArray(strides).join 'x'
    paddingsStr = utils.asArray(paddings).join 'x'
    text = "#{kernelsStr}"
    text += "@#{filters}" if filters?
    text += " + #{stridesStr}s" if strides?
    text += " + #{paddingsStr}p" if paddings?
    text += " kernel_hw" if useKernelHW
    text += " stride_hw" if useStrideHW
    text += " padding_hw" if usePaddingHW
    return text

setupSlidingWindowParameters = (params, kernels, strides, paddings,
                                useKernelHW, useStrideHW, usePaddingHW) ->
    if useKernelHW
        getKernel = (i) =>
            if kernels?[i]? then kernels[i] else kernels
        params.kernel_h = getKernel 0
        params.kernel_w = getKernel 1
    else
        params.kernel_size = kernels
    if useStrideHW
        getStride = (i) =>
            if strides?[i]? then strides[i] else strides
        params.stride_h = getStride 0
        params.stride_w = getStride 1
    else
        params.stride = strides if strides?
    if usePaddingHW
        getPaddings = (i) =>
            if paddings?[i]? then paddings[i] else paddings
        params.pad_h = getPaddings 0
        params.pad_w = getPaddings 1
    else
        params.pad = paddings if paddings?

runConvTasksBase = (tasks, LayerType, useKernelHW,
                    useStrideHW, usePaddingHW) ->
    makeCaffeConvParams = (filters, kernels, strides, paddings) =>
        params =  { num_output: filters }
        setupSlidingWindowParameters params, kernels, strides, paddings,
                                     useKernelHW, useStrideHW, usePaddingHW
        return { convolution_param: params }
    makeConvTaskName = (task) =>
        convParamsStr = stringifyConvParams task[2]...,
                        useKernelHW, useStrideHW, usePaddingHW
        return "from [ #{task[0]} ] to [ #{task[1]} ] with #{convParamsStr}"
    compareConvOutput = (task) =>
        compareLayerOutput LayerType, makeCaffeConvParams, task
    runLayerTasks tasks, makeConvTaskName, compareConvOutput

runConvTasks = (tasks, useKernelHW=false, useStrideHW=false, usePaddingHW=false) ->
    runConvTasksBase tasks, layers.ConvolutionLayer, useKernelHW, useStrideHW, usePaddingHW

runDeconvTasks = (tasks, useKernelHW=false, useStrideHW=false, usePaddingHW=false) ->
    runConvTasksBase tasks, layers.DeconvolutionLayer, useKernelHW, useStrideHW, usePaddingHW

runPoolTasks = (tasks, useKernelHW=false, useStrideHW=false, usePaddingHW=false) ->
    makeCaffePoolParams = (kernels, strides, paddings) ->
        params = { }
        setupSlidingWindowParameters params, kernels, strides, paddings,
                                     useKernelHW, useStrideHW, usePaddingHW
        return { pooling_param: params }
    makePoolTaskName = (task) ->
        poolParamsStr = stringifyConvParams null, task[2]...,
                        useKernelHW, useStrideHW, usePaddingHW
        return "from [ #{task[0]} ] to [ #{task[1]} ] with #{poolParamsStr}"
    comparePoolOutput = (task) ->
        compareLayerOutput layers.PoolingLayer, makeCaffePoolParams, task
    runLayerTasks tasks, makePoolTaskName, comparePoolOutput


runInnerProductTasks = (tasks) ->
    makeCaffeInnerProductParams = (numOutput, axis) ->
        params = { num_output: numOutput }
        params.axis = axis if axis?
        return { inner_product_param: params }
    compareInnerProductOutput = (task) ->
        compareLayerOutput layers.InnerProductLayer, makeCaffeInnerProductParams, task
    makeInnerProductTaskName = (task) ->
        params = task[2]
        text = "from [ #{task[0]} ] to [ #{task[1]} ] where n = #{params[0]}"
        text += " and axis = #{params[1]}" if params[1]?
        return text
    runLayerTasks tasks, makeInnerProductTaskName, compareInnerProductOutput

runConcatTasks = (tasks) ->
    makeCaffeConcatParams = (axis) ->
        params = { }
        params.axis = axis if axis?
        return { concat_param: params }
    compareConcatOutput = (task) ->
        compareLayerOutput layers.ConcatLayer, makeCaffeConcatParams, task
    makeConcatTaskName = (task) ->
        [inputShapes, outputShape, axis] = task
        text = 'from ['
        for shape in inputShapes
            text += " [ #{shape} ]"
        text += " ] to #{outputShape}"
        text += " where axis = #{axis}" if axis?
        return text
    runLayerTasks tasks, makeConcatTaskName, compareConcatOutput

runCropTasks = (tasks) ->
    makeCaffeCropParams = (axis) ->
        params = { }
        params.axis = axis if axis?
        return { crop_param: params }
    compareCropOutput = (task) ->
        compareLayerOutput layers.CropLayer, makeCaffeCropParams, task
    makeCropTaskName = (task) ->
        [inputShapes, outputShape, axis] = task
        text = 'from ['
        for shape in inputShapes
            text += " [ #{shape} ]"
        text += " ] to #{outputShape}"
        text += " where axis = #{axis}" if axis?
        return text
    runLayerTasks tasks, makeCropTaskName, compareCropOutput

runSplitTasks = (tasks) ->
    makeCaffeSplitParams = () ->
        return null
    compareSplitOutput = (task) ->
        compareLayerOutput layers.SplitLayer, makeCaffeSplitParams, task
    makeSplitTaskName = (task) ->
        [inputShape, outputShape] = task
        text = "from [ #{inputShape} ]"
        text += " to #{outputShape}"
        return text
    runLayerTasks tasks, makeSplitTaskName, compareSplitOutput

runAccuracyTasks = (tasks) ->
    makeCaffeAccuracyParams = (axis) ->
        params = { }
        params.axis = axis if axis?
        return { accuracy_param: params }
    compareAccuracyOutput = (task) ->
        compareLayerOutput layers.AccuracyLayer, makeCaffeAccuracyParams, task
    makeAccuracyTaskName = (task) ->
        [inputShapes, outputShapes, axis] = task
        text = 'from ['
        for shape in inputShapes
            text += " [ #{shape} ]"
        if outputShapes.length == 1
            text += " ] to #{outputShapes}"
        else
            text += ' ] to ['
            for out_shape in outputShapes
                text += " #{out_shape}"
            text += ' ] '
        text += " where axis = #{axis}" if axis?
        return text
    runLayerTasks tasks, makeAccuracyTaskName, compareAccuracyOutput

runPermuteTasks = (tasks) ->
    makeCaffePermuteParams = (orders) ->
        params = { }
        params.order = orders if orders?
        return { permute_param: params }
    comparePermuteOutput = (task) ->
        compareLayerOutput layers.PermuteLayer, makeCaffePermuteParams, task
    makePermuteTaskName = (task) ->
        [inputShape, outputShape, orders] = task
        text = "from [ [ #{inputShape} ] ] to #{outputShape}"
        text += " where orders = #{orders}" if orders?
        return text
    runLayerTasks tasks, makePermuteTaskName, comparePermuteOutput

runFlattenTasks = (tasks) ->
    makeCaffeFlattenParams = (axis, end_axis) ->
        params = { }
        params.axis = axis if axis?
        params.end_axis = end_axis if end_axis?
        return { flatten_param: params }
    compareFlattenOutput = (task) ->
        compareLayerOutput layers.FlattenLayer, makeCaffeFlattenParams, task
    makeFlattenTaskName = (task) ->
        params = task[2]
        text = "from [ #{task[0]} ] to [ #{task[1]} ]"
        text += " where axis = #{params[0]}" if params[0]?
        text += " and end_axis = #{params[1]}" if params[1]?
        return text
    runLayerTasks tasks, makeFlattenTaskName, compareFlattenOutput

runPriorBoxTasks = (tasks) ->
    makeCaffePriorBoxParams = (min_size, max_size, aspect_ratio, flip) ->
        params = { }
        params.min_size = min_size if min_size?
        params.max_size = max_size if max_size?
        params.aspect_ratio = aspect_ratio if aspect_ratio?
        params.flip = flip if flip?
        return { prior_box_param: params }
    comparePriorBoxOutput = (task) ->
        compareLayerOutput layers.PriorBoxLayer, makeCaffePriorBoxParams, task
    makePriorBoxTaskName = (task) ->
        params = task[2]
        text = "from [ #{task[0]} ] to [ #{task[1]} ]"
        text += " where min_size = #{params[0]}" if params[0]?
        text += " and max_size = #{params[1]}" if params[1]?
        text += " and aspect_ratio = #{params[2]}" if params[2]?
        text += " and flip = #{params[3]}" if params[3]?
        return text
    runLayerTasks tasks, makePriorBoxTaskName, comparePriorBoxOutput

runReshapeTasks = (tasks) ->
    makeCaffeReshapeParams = (shape, axis, num_axes) ->
        params = { }
        params.shape = { }
        params.shape.dim = shape if shape?
        params.axis = axis if axis?
        params.num_axes = num_axes if num_axes?
        return { reshape_param: params }
    compareReshapeOutput = (task) ->
        compareLayerOutput layers.ReshapeLayer, makeCaffeReshapeParams, task
    makeReshapeTaskName = (task) ->
        params = task[2]
        text = "from [ #{task[0]} ] to [ #{task[1]} ]"
        text += " where shape = #{params[0]}" if params[0]?
        text += " and axis = #{params[1]}" if params[1]?
        text += " and num_axes = #{params[2]}" if params[2]?
        return text
    runLayerTasks tasks, makeReshapeTaskName, compareReshapeOutput

runDetectionOutputTasks = (tasks) ->
    makeCaffeDetectionOutputParams = (num_classes, share_location, keep_top_k) ->
        params = { }
        params.num_classes = num_classes if num_classes?
        params.share_location = share_location if share_location?
        params.keep_top_k = keep_top_k if keep_top_k?
        return { detection_output_param: params }
    compareDetectionOutputOutput = (task) ->
        compareLayerOutput layers.DetectionOutputLayer, makeCaffeDetectionOutputParams, task
    makeDetectionOutputTaskName = (task) ->
        [inputShapes, outputShapes, params] = task
        text = 'from ['
        for shape in inputShapes
            text += " [ #{shape} ]"
        text += " ] to [ #{outputShapes} ]"
        text += " where num_classes = #{params[0]}" if params[0]?
        text += " and share_location = #{params[1]}" if params[1]?
        text += " and keep_top_k = #{params[2]}" if params[2]?
        return text
    runLayerTasks tasks, makeDetectionOutputTaskName, compareDetectionOutputOutput

runArgMaxTasks = (tasks) ->
    makeCaffeArgMaxParams = (axis, top_k, out_max_val) ->
        params = { }
        params.axis = axis if axis?
        params.top_k = top_k if top_k?
        params.out_max_val = out_max_val if out_max_val?
        return { argmax_param: params }
    compareArgMaxOutput = (task) ->
        compareLayerOutput layers.ArgMaxLayer, makeCaffeArgMaxParams, task
    makeArgMaxTaskName = (task) ->
        params = task[2]
        text = "from [ #{task[0]} ] to [ #{task[1]} ]"
        text += " where axis = #{params[0]}" if params[0]?
        text += " and top_k = #{params[1]}" if params[1]?
        text += " and out_max_val = #{params[2]}" if params[2]?
        return text
    runLayerTasks tasks, makeArgMaxTaskName, compareArgMaxOutput

runUpsampleTasks = (tasks) ->
    makeCaffeUpsampleParams = (scale, pad_out_h, pad_out_w) ->
        params = { }
        params.scale = scale if scale?
        params.pad_out_h = pad_out_h if pad_out_h?
        params.pad_out_w = pad_out_w if pad_out_w?
        return { upsample_param: params }
    compareUpsampleOutput = (task) ->
        compareLayerOutput layers.UpsampleLayer, makeCaffeUpsampleParams, task
    makeUpsampleTaskName = (task) ->
        [inputShapes, outputShapes, params] = task
        text = 'from ['
        for shape in inputShapes
            text += " [ #{shape} ]"
        text += " ] to [ #{outputShapes} ]"
        text += " where scale = #{params[0]}" if params[0]?
        text += " and pad_out_h = #{params[1]}" if params[1]?
        text += " and pad_out_w = #{params[2]}" if params[2]?
        return text
    runLayerTasks tasks, makeUpsampleTaskName, compareUpsampleOutput

runInterpTasks = (tasks) ->
    makeCaffeInterpParams = (height, width) ->
        params = { }
        params.height = height if height?
        params.width = width if width?
        return { interp_param: params }
    compareInterpOutput = (task) ->
        compareLayerOutput layers.InterpLayer, makeCaffeInterpParams, task
    makeInterpTaskName = (task) ->
        params = task[2]
        text = "from [ #{task[0]} ] to [ #{task[1]} ]"
        text += " where height = #{params[0]}" if params[0]?
        text += " and width = #{params[1]}" if params[1]?
        return text
    runLayerTasks tasks, makeInterpTaskName, compareInterpOutput

runSliceTasks = (tasks) ->
    makeCaffeSliceParams = (slice_dim, axis, slice_point) ->
        params = { }
        if axis?
            params.axis = axis
            params.slice_point = slice_point if slice_point?
        else
            params.slice_dim = slice_dim if slice_dim?
        return { slice_param: params }
    compareSliceOutput = (task) ->
        compareLayerOutput layers.SliceLayer, makeCaffeSliceParams, task
    makeSliceTaskName = (task) ->
        [inputShapes, outputShapes, params] = task
        text = "from [ #{task[0]} ] to ["
        for shape in outputShapes
            text += " [ #{shape} ]"
        text += " ] where slice_dim = #{params[0]}" if params[0]?
        text += " and axis = #{params[1]}" if params[1]?
        text += " and slice_point = #{params[2]}" if params[2]?
        return text
    runLayerTasks tasks, makeSliceTaskName, compareSliceOutput

describe 'Compute 2D Convolution output shape', ->
    # [ input shape, expecting output shape ]
    # null means default parameter value
    shapes1 = (p) -> [ [32, 3, 227, 227], [32, 96, 55, 55], p ]
    shapes2 = (p) -> [ [32, 256, 27, 27], [32, 256, 27, 27], p ]
    shapes3 = (p) -> [ [1, 256, 15, 15], [1, 96, 15, 15], p ]
    # [filters, kernels, strides, paddings]
    tasks = [
        shapes1 [ 96,  [11, 11], [4, 4], [0, 0] ]
        shapes1 [ 96,  [11, 11], [4, 4],   0    ]
        shapes1 [ 96,  [11, 11], [4, 4],  null  ]
        shapes1 [ 96,  [11, 11],    4,   [0, 0] ]
        shapes1 [ 96,  [11, 11],    4,     0    ]
        shapes1 [ 96,  [11, 11],    4,    null  ]
        shapes1 [ 96,     11,    [4, 4], [0, 0] ]
        shapes1 [ 96,     11,    [4, 4],   0    ]
        shapes1 [ 96,     11,    [4, 4],  null  ]
        shapes1 [ 96,     11,       4,   [0, 0] ]
        shapes1 [ 96,     11,       4,     0    ]
        shapes1 [ 96,     11,       4,    null  ]
        shapes2 [ 256, [5, 5],   [1, 1], [2, 2] ]
        shapes2 [ 256, [5, 5],   [1, 1],   2    ]
        shapes2 [ 256, [5, 5],     1,    [2, 2] ]
        shapes2 [ 256, [5, 5],     1,      2    ]
        shapes2 [ 256,    5,     [1, 1], [2, 2] ]
        shapes2 [ 256,    5,     [1, 1],   2    ]
        shapes2 [ 256,    5,       1,    [2, 2] ]
        shapes2 [ 256,    5,       1,      2    ]
        shapes2 [ 256,    5,      null,  [2, 2] ]
        shapes2 [ 256,    5,      null,    2    ]
        shapes2 [ 256, [5, 5],    null,  [2, 2] ]
        shapes2 [ 256, [5, 5],    null,    2    ]
        shapes3 [ 96,  [1, 7],   [1, 1], [0, 3] ]
    ]
    falsetrue = [false, true]
    for useKernelHW in falsetrue
        for useStrideHW in falsetrue
            for usePaddingHW in falsetrue
                runConvTasks tasks, useKernelHW, useStrideHW, usePaddingHW

describe 'Compute 2D Deconvolution output shape', ->
    # [ input shape, expecting output shape ]
    # null means default parameter value
    shapes1 = (p) -> [ [32, 96, 55, 55], [32, 3, 227, 227], p ]
    shapes2 = (p) -> [ [32, 256, 27, 27], [32, 256, 27, 27], p ]
    shapes3 = (p) -> [ [1, 96, 15, 15], [1, 256, 15, 15], p ]
    # [filters, kernels, strides, paddings]
    tasks = [
        shapes1 [ 3,  [11, 11], [4, 4], [0, 0] ]
        shapes1 [ 3,  [11, 11], [4, 4],   0    ]
        shapes1 [ 3,  [11, 11], [4, 4],  null  ]
        shapes1 [ 3,  [11, 11],    4,   [0, 0] ]
        shapes1 [ 3,  [11, 11],    4,     0    ]
        shapes1 [ 3,  [11, 11],    4,    null  ]
        shapes1 [ 3,     11,    [4, 4], [0, 0] ]
        shapes1 [ 3,     11,    [4, 4],   0    ]
        shapes1 [ 3,     11,    [4, 4],  null  ]
        shapes1 [ 3,     11,       4,   [0, 0] ]
        shapes1 [ 3,     11,       4,     0    ]
        shapes1 [ 3,     11,       4,    null  ]
        shapes2 [ 256, [5, 5],   [1, 1], [2, 2] ]
        shapes2 [ 256, [5, 5],   [1, 1],   2    ]
        shapes2 [ 256, [5, 5],     1,    [2, 2] ]
        shapes2 [ 256, [5, 5],     1,      2    ]
        shapes2 [ 256,    5,     [1, 1], [2, 2] ]
        shapes2 [ 256,    5,     [1, 1],   2    ]
        shapes2 [ 256,    5,       1,    [2, 2] ]
        shapes2 [ 256,    5,       1,      2    ]
        shapes2 [ 256,    5,      null,  [2, 2] ]
        shapes2 [ 256,    5,      null,    2    ]
        shapes2 [ 256, [5, 5],    null,  [2, 2] ]
        shapes2 [ 256, [5, 5],    null,    2    ]
        shapes3 [ 256,  [1, 7],   [1, 1], [0, 3] ]
    ]
    falsetrue = [false, true]
    for useKernelHW in falsetrue
        for useStrideHW in falsetrue
            for usePaddingHW in falsetrue
                runDeconvTasks tasks, useKernelHW, useStrideHW, usePaddingHW

describe 'Compute 3D Convolution output shape', ->
    # [ input shape, expecting output shape ]
    shapes1 = (p) -> [ [1, 3, 224, 224, 224], [1, 64, 112, 112, 112], p ]
    shapes2 = (p) -> [ [1, 64, 28, 28, 28], [1, 128, 14, 14, 14], p ]
    # [filters, kernels, strides, paddings]
    tasks = [
        shapes1 [ 64, [7, 7, 7], [2, 2, 2], [3, 3, 3] ]
        shapes1 [ 64,     7,         2,         3     ]
        shapes1 [ 64, [2, 7, 7],     2,     [0, 3, 3] ]
        shapes1 [ 64, [7, 2, 7],     2,     [3, 0, 3] ]
        shapes1 [ 64, [7, 7, 2],     2,     [3, 3, 0] ]
        shapes2 [ 128, [7, 7, 2], 2, [3, 3, 0] ]
    ]
    runConvTasks tasks

describe 'Compute Pooling output shape', ->
    # [ input shape, expecting output shape ]
    shapes1 = (p) -> [ [1, 192, 56, 56], [1, 192, 28, 28], p ]
    shapes2 = (p) -> [ [1, 192, 28, 28], [1, 192, 28, 28], p ]
    # [ kernels, strides, paddings ]
    tasks = [
        shapes1 [ [3, 3], [2, 2], [0, 0] ]
        shapes1 [ [3, 3], [2, 2],    0   ]
        shapes1 [ [3, 3], [2, 2],  null  ]
        shapes1 [ [3, 3],    2,   [0, 0] ]
        shapes1 [ [3, 3],    2,      0   ]
        shapes1 [ [3, 3],    2,    null  ]
        shapes1 [   3,    [2, 2], [0, 0] ]
        shapes1 [   3,    [2, 2],    0   ]
        shapes1 [   3,    [2, 2],  null  ]
        shapes1 [   3,       2,   [0, 0] ]
        shapes1 [   3,       2,      0   ]
        shapes1 [   3,       2,    null  ]
        shapes2 [ [3, 3], [1, 1], [1, 1] ]
        shapes2 [ [3, 3], [1, 1],    1   ]
        shapes2 [ [3, 3],    1,   [1, 1] ]
        shapes2 [ [3, 3],    1,      1   ]
        shapes2 [ [3, 3],  null,  [1, 1] ]
        shapes2 [ [3, 3],  null,     1   ]
    ]
    falsetrue = [false, true]
    for useKernelHW in falsetrue
        for useStrideHW in falsetrue
            for usePaddingHW in falsetrue
                runPoolTasks tasks, useKernelHW, useStrideHW, usePaddingHW

describe 'Compute InnerProduct output shape', ->
    # [ input shape, expecting output shape, [ num_outputs, axis ] ]
    tasks = [
        [ [32, 300, 28], [ 128 ], [128, 0] ]
        [ [32, 300, 28, 28], [32, 512], [512] ]
        [ [32, 300, 28, 28], [32, 512], [512, 1] ]
        [ [32, 300, 28, 28, 46], [32, 1024], [1024] ]
        [ [32, 300, 28, 28, 46], [32, 1024], [1024, 1] ]
        [ [32, 300, 28, 28, 46], [32, 300, 1024], [1024, 2] ]
        [ [32, 300, 28, 28, 46], [32, 300, 28, 512], [512, 3] ]
    ]
    runInnerProductTasks tasks

describe 'Compute Concat output shape', ->
    # [ [ input shapes ], expecting output shape, axis ]
    tasks = [
        [ [[32, 54, 43, 43]], [32, 54, 43, 43], null ]
        [ [[32, 54, 43, 43], [32, 21, 43, 43]], [32, 75, 43, 43], null ]
        [ [[32, 21, 43, 43], [32, 21, 43, 43]], [64, 21, 43, 43], 0 ]
        [ [[32, 54, 43, 43], [32, 21, 43, 43]], [32, 75, 43, 43], 1 ]
        [ [[32, 21, 43, 43], [32, 21, 20, 43]], [32, 21, 63, 43], 2 ]
        [ [[32, 21, 30, 30], [32, 21, 30, 25]], [32, 21, 30, 55], 3 ]
    ]
    runConcatTasks tasks

describe 'Compute Crop output shape', ->
    # [ [bottom[0] shape, bottom[1] shape], expecting output shape, axis ]
    tasks = [
        [ [[1, 21, 44, 44], [1, 21, 34, 34]], [1, 21, 34, 34] ]
        [ [[1, 21, 88, 88], [1, 21, 70, 70]], [1, 21, 70, 70] ]
        [ [[1, 21, 44, 44], [1, 21, 34, 34]], [1, 21, 34, 34], 1 ]
        [ [[1, 21, 88, 88], [1, 21, 70, 70]], [1, 21, 70, 70], 0 ]
        [ [[1, 21, 568, 568], [1, 3, 500, 500]], [1, 3, 500, 500] ]
        [ [[64, 32, 15, 19], [64, 16, 15, 15]], [64, 16, 15, 15], 1 ]
        [ [[64, 32, 20, 32], [64, 16, 15, 30]], [64, 32, 15, 30], 2 ]
        [ [[64, 32, 20, 32], [64, 16, 15, 30]], [64, 32, 20, 30], 3 ]
    ]
    runCropTasks tasks

describe 'Compute Split output shape', ->
    # [ input shape, expecting output shape ]
    tasks = [
        [ [1, 21, 44, 44], [1, 21, 44, 44] ]
    ]
    runSplitTasks tasks

describe 'Compute Accuracy output shape', ->
    # [ [bottom[0] shape, bottom[1] shape], expecting output shape, axis ]
    tasks = [
        [ [[1, 1000], [1000]], [1] ]
        [ [[1, 1000], [1000]], [1], 1 ]
        [ [[1, 1000], [1000]], [[1], [1000]], 1 ]
    ]
    runAccuracyTasks tasks

describe 'Compute Permute output shape', ->
    # [ input shape, expecting output shape, orders ]
    tasks = [
        [ [1, 8, 64, 128], [1, 64, 128, 8], [ [0, 2, 3, 1] ] ]
        [ [1, 8, 64, 128], [1, 64, 8, 128], [ [0, 2] ] ]
    ]
    runPermuteTasks tasks

describe 'Compute Flatten output shape', ->
    # [ input shape, expecting output shape, [axis, end_axis] ]
    tasks = [
        [ [1, 8, 10, 10], [1, 800], [1] ]
        [ [1, 8, 10, 10], [1, 80, 10], [1, -2] ]
    ]
    runFlattenTasks tasks

describe 'Compute PriorBox output shape', ->
    # [ [bottom[0] shape, bottom[1] shape], expecting output shape, [ min_size, max_size, aspect_ratio, flip ] ]
    tasks = [
        [ [[1, 32, 10, 10], [1, 3, 224, 224]], [ 1, 2, 2400 ], [105.0, 150.0, [2.0, 3.0], 'true'] ]
        [ [[1, 32, 10, 10], [1, 3, 224, 224]], [ 1, 2, 2800 ], [105.0, [150.0, 160.0], [2.0, 3.0], 'true'] ]
        [ [[1, 32, 10, 10], [1, 3, 224, 224]], [ 1, 2, 1600 ], [105.0, 150.0, [2.0, 3.0], 'false'] ]
    ]
    runPriorBoxTasks tasks

describe 'Compute Reshape output shape', ->
    # [ input shape, expecting output shape, [ shape, axis, num_axes ] ]
    tasks = [
        [ [2, 8], [2, 2, 4], [ [0, -1, 4] ] ]
        [ [1, 40257], [1, 1917, 21], [ [0, -1, 21] ] ]
    ]
    runReshapeTasks tasks

describe 'Compute DetectionOutput output shape', ->
    # [ [bottom[0] shape, bottom[1] shape, bottom[2] shape], expecting output shape, [ num_classes, share_location, keep_top_k ] ]
    tasks = [
        [ [[1, 7668], [1, 40257], [1, 2, 7668]], [ 100, 7 ], [21, 'true', 100] ]
    ]
    runDetectionOutputTasks tasks

describe 'Compute ArgMax output shape', ->
    # [ bottom[0] shape, expecting output shape, [ axis, top_k, out_max_val ] ]
    tasks = [
        [ [1, 21, 256, 512], [ 1, 1, 256, 512 ], [1] ]
        [ [1, 21, 256, 512], [ 1, 5, 256, 512 ], [1, 5] ]
    ]
    runArgMaxTasks tasks

describe 'Compute Upsample output shape', ->
    # [ [bottom[0] shape, bottom[1] shape], expecting output shape, [ scale, pad_out_h, pad_out_w ] ]
    tasks = [
        [ [[1, 4, 5, 5], [1, 4, 5, 5]], [1, 4, 9, 10], [2, 'true', 'false'] ]
    ]
    runUpsampleTasks tasks

describe 'Compute Interp output shape', ->
    # [ bottom[0] shape, expecting output shape, [ height, width ] ]
    tasks = [
        [ [1, 4, 5, 5], [1, 4, 15, 25], [15, 25] ]
    ]
    runInterpTasks tasks

describe 'Compute Slice output shape', ->
    # [ bottom[0] shape, expecting output shape, [ slice_dim axis slice_point slice_point ] ]
    tasks = [
        [ [1, 4, 5, 5], [[1, 2, 5, 5], [1, 2, 5, 5]], [1] ]
        [ [1, 6, 5, 5], [[1, 1, 5, 5], [1, 3, 5, 5], [1, 2, 5, 5]], [1, -3, [1, 4]] ]
    ]
    runSliceTasks tasks