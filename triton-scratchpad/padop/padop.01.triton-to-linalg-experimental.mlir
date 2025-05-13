// -----// IR Dump After TritonToLinalgExperimental Failed (triton-to-linalg-experimental) ('builtin.module' operation) //----- //
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (0, d1)>
#map3 = affine_map<() -> (0)>
#map4 = affine_map<(d0, d1) -> (d0, 0)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{maia.rank = 1 : i32, tt.divisibility = 16 : i32}, {maia.rank = 1 : i32, tt.divisibility = 16 : i32}, {maia.rank = 3 : i32, tt.divisibility = 16 : i32}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}], function_type = (memref<*xi64>, memref<*xi32>, memref<*xf32>, i32, i32, i32, i32, i32, f32, i32, i32, i32, i32, i32, i32) -> (), sym_name = "pad_sequence_kernel"}> ({
  ^bb0(%arg0: memref<*xi64>, %arg1: memref<*xi32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: f32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32):
    %0 = "arith.constant"() <{value = 128 : index}> : () -> index
    %1 = "arith.constant"() <{value = 1 : index}> : () -> index
    %2 = "arith.constant"() <{value = 127 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 128 : i32}> : () -> i32
    %4 = "arith.addi"(%arg5, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %5 = "arith.divsi"(%4, %3) : (i32, i32) -> i32
    %6 = "arith.muli"(%5, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %7 = "arith.muli"(%arg13, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %8 = "arith.index_cast"(%7) : (i32) -> index
    %9 = "tensor.empty"() : () -> tensor<128xi32>
    %10 = "linalg.generic"(%9) <{indexing_maps = [#map], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb0(%arg72: i32):
      %92 = "linalg.index"() <{dim = 0 : i64}> : () -> index
      %93 = "arith.index_cast"(%92) : (index) -> i32
      "linalg.yield"(%93) : (i32) -> ()
    }) : (tensor<128xi32>) -> tensor<128xi32>
    %11 = "linalg.fill"(%7, %9) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg70: i32, %arg71: i32):
      "linalg.yield"(%arg70) : (i32) -> ()
    }) : (i32, tensor<128xi32>) -> tensor<128xi32>
    %12 = "linalg.generic"(%11, %10, %11) <{indexing_maps = [#map, #map, #map], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg67: i32, %arg68: i32, %arg69: i32):
      %91 = "arith.addi"(%arg67, %arg68) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "linalg.yield"(%91) : (i32) -> ()
    }) : (tensor<128xi32>, tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
    %13 = "tensor.expand_shape"(%12) <{reassociation = [[0, 1]], static_output_shape = array<i64: 128, 1>}> : (tensor<128xi32>) -> tensor<128x1xi32>
    %14 = "arith.muli"(%arg14, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %15 = "arith.index_cast"(%14) : (i32) -> index
    %16 = "linalg.fill"(%14, %9) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg65: i32, %arg66: i32):
      "linalg.yield"(%arg65) : (i32) -> ()
    }) : (i32, tensor<128xi32>) -> tensor<128xi32>
    %17 = "linalg.generic"(%16, %10, %16) <{indexing_maps = [#map, #map, #map], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg62: i32, %arg63: i32, %arg64: i32):
      %90 = "arith.addi"(%arg62, %arg63) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "linalg.yield"(%90) : (i32) -> ()
    }) : (tensor<128xi32>, tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
    %18 = "tensor.expand_shape"(%17) <{reassociation = [[0, 1]], static_output_shape = array<i64: 1, 128>}> : (tensor<128xi32>) -> tensor<1x128xi32>
    %19 = "tensor.empty"() : () -> tensor<1x128xi32>
    %20 = "linalg.fill"(%arg4, %19) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg60: i32, %arg61: i32):
      "linalg.yield"(%arg60) : (i32) -> ()
    }) : (i32, tensor<1x128xi32>) -> tensor<1x128xi32>
    %21 = "tensor.empty"() : () -> tensor<1x128xi1>
    %22 = "linalg.generic"(%18, %20, %21) <{indexing_maps = [#map1, #map1, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg57: i32, %arg58: i32, %arg59: i1):
      %89 = "arith.cmpi"(%arg57, %arg58) <{predicate = 2 : i64}> : (i32, i32) -> i1
      "linalg.yield"(%89) : (i1) -> ()
    }) : (tensor<1x128xi32>, tensor<1x128xi32>, tensor<1x128xi1>) -> tensor<1x128xi1>
    %23 = "tensor.empty"() : () -> tensor<128x128xi1>
    %24 = "linalg.generic"(%22, %23) <{indexing_maps = [#map2, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg55: i1, %arg56: i1):
      "linalg.yield"(%arg55) : (i1) -> ()
    }) {broadcastDims = array<i64: 0>} : (tensor<1x128xi1>, tensor<128x128xi1>) -> tensor<128x128xi1>
    %25 = "arith.muli"(%arg12, %arg7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %26 = "arith.index_cast"(%25) : (i32) -> index
    %27 = "arith.index_cast"(%arg6) : (i32) -> index
    %28 = "arith.muli"(%8, %27) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %29 = "arith.addi"(%26, %28) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %30 = "tensor.empty"() : () -> tensor<128x128xi32>
    %31 = "linalg.generic"(%18, %30) <{indexing_maps = [#map2, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg53: i32, %arg54: i32):
      "linalg.yield"(%arg53) : (i32) -> ()
    }) {broadcastDims = array<i64: 0>} : (tensor<1x128xi32>, tensor<128x128xi32>) -> tensor<128x128xi32>
    %32 = "scf.for"(%arg12, %arg3, %arg9, %29) ({
    ^bb0(%arg15: i32, %arg16: index):
      %33 = "arith.addi"(%arg16, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %34 = "memref.reinterpret_cast"(%arg2, %33, %27, %1) <{operandSegmentSizes = array<i32: 1, 1, 0, 2>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 128, 128>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<*xf32>, index, index, index) -> memref<128x128xf32, strided<[?, ?], offset: ?>>
      %35 = "arith.index_cast"(%arg15) : (i32) -> index
      %36 = "memref.reinterpret_cast"(%arg0, %35) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (memref<*xi64>, index) -> memref<1xi64, strided<[1], offset: ?>>
      %37 = "affine.load"(%36) <{map = #map3}> : (memref<1xi64, strided<[1], offset: ?>>) -> i64
      %38 = "tptr.inttoptr"(%37) : (i64) -> !ptr.ptr
      %39 = "tptr.to_memref"(%38) : (!ptr.ptr) -> memref<f32>
      %40 = "memref.reinterpret_cast"(%arg1, %35) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>}> : (memref<*xi32>, index) -> memref<1xi32, strided<[1], offset: ?>>
      %41 = "affine.load"(%40) <{map = #map3}> : (memref<1xi32, strided<[1], offset: ?>>) -> i32
      %42 = "arith.subi"(%arg5, %41) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %43 = "tensor.empty"() : () -> tensor<128x1xi32>
      %44 = "linalg.fill"(%42, %43) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg51: i32, %arg52: i32):
        "linalg.yield"(%arg51) : (i32) -> ()
      }) : (i32, tensor<128x1xi32>) -> tensor<128x1xi32>
      %45 = "linalg.generic"(%13, %44, %13) <{indexing_maps = [#map1, #map1, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg48: i32, %arg49: i32, %arg50: i32):
        %88 = "arith.subi"(%arg48, %arg49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "linalg.yield"(%88) : (i32) -> ()
      }) : (tensor<128x1xi32>, tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi32>
      %46 = "linalg.fill"(%6, %43) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg46: i32, %arg47: i32):
        "linalg.yield"(%arg46) : (i32) -> ()
      }) : (i32, tensor<128x1xi32>) -> tensor<128x1xi32>
      %47 = "linalg.generic"(%45, %46, %45) <{indexing_maps = [#map1, #map1, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg43: i32, %arg44: i32, %arg45: i32):
        %87 = "arith.addi"(%arg43, %arg44) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "linalg.yield"(%87) : (i32) -> ()
      }) : (tensor<128x1xi32>, tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi32>
      %48 = "linalg.generic"(%47, %46, %47) <{indexing_maps = [#map1, #map1, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg40: i32, %arg41: i32, %arg42: i32):
        %86 = "arith.remsi"(%arg40, %arg41) : (i32, i32) -> i32
        "linalg.yield"(%86) : (i32) -> ()
      }) : (tensor<128x1xi32>, tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi32>
      %49 = "linalg.fill"(%arg4, %43) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg38: i32, %arg39: i32):
        "linalg.yield"(%arg38) : (i32) -> ()
      }) : (i32, tensor<128x1xi32>) -> tensor<128x1xi32>
      %50 = "linalg.generic"(%48, %49, %48) <{indexing_maps = [#map1, #map1, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg35: i32, %arg36: i32, %arg37: i32):
        %85 = "arith.muli"(%arg35, %arg36) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "linalg.yield"(%85) : (i32) -> ()
      }) : (tensor<128x1xi32>, tensor<128x1xi32>, tensor<128x1xi32>) -> tensor<128x1xi32>
      %51 = "linalg.generic"(%50, %30) <{indexing_maps = [#map4, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg33: i32, %arg34: i32):
        "linalg.yield"(%arg33) : (i32) -> ()
      }) {broadcastDims = array<i64: 1>} : (tensor<128x1xi32>, tensor<128x128xi32>) -> tensor<128x128xi32>
      %52 = "linalg.generic"(%51, %31, %51) <{indexing_maps = [#map1, #map1, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg30: i32, %arg31: i32, %arg32: i32):
        %84 = "arith.addi"(%arg30, %arg31) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "linalg.yield"(%84) : (i32) -> ()
      }) : (tensor<128x128xi32>, tensor<128x128xi32>, tensor<128x128xi32>) -> tensor<128x128xi32>
      %53 = "linalg.fill"(%41, %43) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg28: i32, %arg29: i32):
        "linalg.yield"(%arg28) : (i32) -> ()
      }) : (i32, tensor<128x1xi32>) -> tensor<128x1xi32>
      %54 = "tensor.empty"() : () -> tensor<128x1xi1>
      %55 = "linalg.generic"(%48, %53, %54) <{indexing_maps = [#map1, #map1, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg25: i32, %arg26: i32, %arg27: i1):
        %83 = "arith.cmpi"(%arg25, %arg26) <{predicate = 2 : i64}> : (i32, i32) -> i1
        "linalg.yield"(%83) : (i1) -> ()
      }) : (tensor<128x1xi32>, tensor<128x1xi32>, tensor<128x1xi1>) -> tensor<128x1xi1>
      %56 = "linalg.generic"(%55, %23) <{indexing_maps = [#map4, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg23: i1, %arg24: i1):
        "linalg.yield"(%arg23) : (i1) -> ()
      }) {broadcastDims = array<i64: 1>} : (tensor<128x1xi1>, tensor<128x128xi1>) -> tensor<128x128xi1>
      %57 = "linalg.generic"(%56, %24, %56) <{indexing_maps = [#map1, #map1, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg20: i1, %arg21: i1, %arg22: i1):
        %82 = "arith.andi"(%arg20, %arg21) : (i1, i1) -> i1
        "linalg.yield"(%82) : (i1) -> ()
      }) : (tensor<128x128xi1>, tensor<128x128xi1>, tensor<128x128xi1>) -> tensor<128x128xi1>
      %58 = "memref.cast"(%39) : (memref<f32>) -> memref<?xf32>
      %59 = "bufferization.to_tensor"(%58) <{restrict}> : (memref<?xf32>) -> tensor<?xf32>
      %60 = "tensor.empty"() : () -> tensor<128x128xf32>
      %61 = "linalg.generic"(%52, %57, %60) <{indexing_maps = [#map1, #map1, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg17: i32, %arg18: i1, %arg19: f32):
        %79 = "scf.if"(%arg18) ({
          %80 = "arith.index_cast"(%arg17) : (i32) -> index
          %81 = "tensor.extract"(%59, %80) : (tensor<?xf32>, index) -> f32
          "scf.yield"(%81) : (f32) -> ()
        }, {
          "scf.yield"(%arg8) : (f32) -> ()
        }) : (i1) -> f32
        "linalg.yield"(%79) : (f32) -> ()
      }) : (tensor<128x128xi32>, tensor<128x128xi1>, tensor<128x128xf32>) -> tensor<128x128xf32>
      %62 = "arith.addi"(%8, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %63 = "arith.index_cast"(%arg5) : (i32) -> index
      %64 = "arith.minsi"(%62, %63) : (index, index) -> index
      %65 = "arith.maxsi"(%64, %8) : (index, index) -> index
      %66 = "arith.subi"(%65, %8) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %67 = "arith.addi"(%15, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %68 = "arith.index_cast"(%arg4) : (i32) -> index
      %69 = "arith.minsi"(%67, %68) : (index, index) -> index
      %70 = "arith.maxsi"(%69, %15) : (index, index) -> index
      %71 = "arith.subi"(%70, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %72 = "arith.minsi"(%66, %0) : (index, index) -> index
      %73 = "arith.minsi"(%71, %0) : (index, index) -> index
      %74 = "tensor.extract_slice"(%61, %72, %73) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (tensor<128x128xf32>, index, index) -> tensor<?x?xf32>
      %75 = "memref.subview"(%34, %72, %73) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<128x128xf32, strided<[?, ?], offset: ?>>, index, index) -> memref<?x?xf32, strided<[?, ?], offset: ?>>
      "bufferization.materialize_in_destination"(%74, %75) <{writable}> : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
      %76 = "arith.muli"(%arg7, %arg9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %77 = "arith.index_cast"(%76) : (i32) -> index
      %78 = "arith.addi"(%arg16, %77) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "scf.yield"(%78) : (index) -> ()
    }) : (i32, i32, i32, index) -> index
    "func.return"() : () -> ()
  }) : () -> ()
}) {maia.triton_kernel} : () -> ()


