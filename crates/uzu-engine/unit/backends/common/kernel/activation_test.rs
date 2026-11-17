use std::fmt::{Debug, Display};

use futures::executor::block_on;
use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::activation_type::ActivationType, kernel::ActivationKernel,
        },
        cpu::Cpu,
    },
    data_type::{DataType, DataTypeRepr},
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend, for_each_non_cpu_backend},
    },
};

struct Input<T: DataTypeRepr + Float> {
    data: Box<[T]>,
    act_type: ActivationType,
    alpha: f32,
    in_place: bool,
}

fn make_data<T: DataTypeRepr + Float>(n: usize) -> Vec<T> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        // Use a range of values including negative to exercise both branches of activations
        data.push(T::from((i as f32 * 0.3) - 2.0).unwrap());
    }
    data
}

fn get_output<T: DataTypeRepr + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = block_on(B::Context::new()).expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::ActivationKernel::new(&context, T::data_type(), input.in_place)
        .expect("Failed to create ActivationKernel");

    let n = input.data.len();

    if input.in_place {
        let mut output_buffer = create_buffer_with_data::<B, T>(&context, &input.data);
        let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to get command buffer");
        kernel.encode(
            None::<&B::GlobalGpuBuffer>,
            &mut output_buffer,
            n as u32,
            input.act_type,
            input.alpha,
            &mut command_buffer,
        );
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
        buffer_to_vec::<B, T>(&context, &output_buffer)
    } else {
        let input_buffer = create_buffer_with_data::<B, T>(&context, &input.data);
        let mut output_buffer = create_buffer::<B, T>(&context, n);
        let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to get command buffer");
        kernel.encode(
            Some(&input_buffer),
            &mut output_buffer,
            n as u32,
            input.act_type,
            input.alpha,
            &mut command_buffer,
        );
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
        buffer_to_vec::<B, T>(&context, &output_buffer)
    }
}

fn get_test_data<T: DataTypeRepr + Float>(
    act_type: ActivationType,
    alpha: f32,
    in_place: bool,
) -> (Input<T>, Vec<T>) {
    let n = 128;
    let data = make_data::<T>(n);

    let input = Input {
        data: data.into_boxed_slice(),
        act_type,
        alpha,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

/// Large input to exercise more threads on GPU.
fn get_test_data_large<T: DataTypeRepr + Float>(
    act_type: ActivationType,
    alpha: f32,
    in_place: bool,
) -> (Input<T>, Vec<T>) {
    let n = 4096;
    let data = make_data::<T>(n);

    let input = Input {
        data: data.into_boxed_slice(),
        act_type,
        alpha,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn test_internal<T: DataTypeRepr + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
    test_name: &str,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let actual = get_output::<T, B>(input);
        let msg = format!("Activation {} failed with backend={}", test_name, std::any::type_name::<B>(),);
        assert_eq_float::<T>(expected, &actual, eps, &msg);
    });
}

fn test_activation<T: DataTypeRepr + Float + Debug + Display>(
    act_type: ActivationType,
    alpha: f32,
    in_place: bool,
) {
    let label = format!(
        "{:?}_alpha{}_{}",
        act_type,
        alpha,
        if in_place {
            "in_place"
        } else {
            "out_of_place"
        }
    );
    let (input, expected) = get_test_data::<T>(act_type, alpha, in_place);
    test_internal::<T>(&input, &expected, &label);
}

fn test_activation_large<T: DataTypeRepr + Float + Debug + Display>(
    act_type: ActivationType,
    alpha: f32,
    in_place: bool,
) {
    let label = format!(
        "{:?}_alpha{}_{}_large",
        act_type,
        alpha,
        if in_place {
            "in_place"
        } else {
            "out_of_place"
        }
    );
    let (input, expected) = get_test_data_large::<T>(act_type, alpha, in_place);
    test_internal::<T>(&input, &expected, &label);
}

// SILU out-of-place tests
#[uzu_test]
fn test_silu_f32() {
    test_activation::<f32>(ActivationType::SILU, 1.0, false);
}

#[uzu_test]
fn test_silu_f16() {
    test_activation::<f16>(ActivationType::SILU, 1.0, false);
}

#[uzu_test]
fn test_silu_bf16() {
    test_activation::<bf16>(ActivationType::SILU, 1.0, false);
}

// SILU in-place tests
#[uzu_test]
fn test_silu_in_place_f32() {
    test_activation::<f32>(ActivationType::SILU, 1.0, true);
}

#[uzu_test]
fn test_silu_in_place_f16() {
    test_activation::<f16>(ActivationType::SILU, 1.0, true);
}

#[uzu_test]
fn test_silu_in_place_bf16() {
    test_activation::<bf16>(ActivationType::SILU, 1.0, true);
}

// GELU out-of-place tests
#[uzu_test]
fn test_gelu_f32() {
    test_activation::<f32>(ActivationType::GELUApprox, 1.0, false);
}

#[uzu_test]
fn test_gelu_f16() {
    test_activation::<f16>(ActivationType::GELUApprox, 1.0, false);
}

#[uzu_test]
fn test_gelu_bf16() {
    test_activation::<bf16>(ActivationType::GELUApprox, 1.0, false);
}

// Exact GELU out-of-place tests
#[uzu_test]
fn test_gelu_exact_f32() {
    test_activation::<f32>(ActivationType::GELUExact, 1.0, false);
}

#[uzu_test]
fn test_gelu_exact_f16() {
    test_activation::<f16>(ActivationType::GELUExact, 1.0, false);
}

#[uzu_test]
fn test_gelu_exact_bf16() {
    test_activation::<bf16>(ActivationType::GELUExact, 1.0, false);
}

// Sigmoid out-of-place tests
#[uzu_test]
fn test_sigmoid_f32() {
    test_activation::<f32>(ActivationType::SIGMOID, 1.0, false);
}

#[uzu_test]
fn test_sigmoid_f16() {
    test_activation::<f16>(ActivationType::SIGMOID, 1.0, false);
}

#[uzu_test]
fn test_sigmoid_bf16() {
    test_activation::<bf16>(ActivationType::SIGMOID, 1.0, false);
}

// GELU in-place tests
#[uzu_test]
fn test_gelu_in_place_f32() {
    test_activation::<f32>(ActivationType::GELUApprox, 1.0, true);
}

#[uzu_test]
fn test_gelu_in_place_f16() {
    test_activation::<f16>(ActivationType::GELUApprox, 1.0, true);
}

#[uzu_test]
fn test_gelu_in_place_bf16() {
    test_activation::<bf16>(ActivationType::GELUApprox, 1.0, true);
}

// Large SILU tests
#[uzu_test]
fn test_silu_large_f32() {
    test_activation_large::<f32>(ActivationType::SILU, 1.0, false);
}

#[uzu_test]
fn test_silu_large_f16() {
    test_activation_large::<f16>(ActivationType::SILU, 1.0, false);
}

#[uzu_test]
fn test_silu_large_bf16() {
    test_activation_large::<bf16>(ActivationType::SILU, 1.0, false);
}

// Large GELU tests
#[uzu_test]
fn test_gelu_large_f32() {
    test_activation_large::<f32>(ActivationType::GELUApprox, 1.0, false);
}

#[uzu_test]
fn test_gelu_large_f16() {
    test_activation_large::<f16>(ActivationType::GELUApprox, 1.0, false);
}

#[uzu_test]
fn test_gelu_large_bf16() {
    test_activation_large::<bf16>(ActivationType::GELUApprox, 1.0, false);
}

// Large in-place tests
#[uzu_test]
fn test_silu_in_place_large_f32() {
    test_activation_large::<f32>(ActivationType::SILU, 1.0, true);
}

#[uzu_test]
fn test_gelu_in_place_large_f32() {
    test_activation_large::<f32>(ActivationType::GELUApprox, 1.0, true);
}

// SILU with custom alpha tests
#[uzu_test]
fn test_silu_custom_alpha_f32() {
    test_activation::<f32>(ActivationType::SILU, 1.7, false);
}

#[uzu_test]
fn test_silu_custom_alpha_f16() {
    test_activation::<f16>(ActivationType::SILU, 1.7, false);
}

#[uzu_test]
fn test_silu_custom_alpha_bf16() {
    test_activation::<bf16>(ActivationType::SILU, 1.7, false);
}

#[uzu_test]
fn test_silu_custom_alpha_in_place_f32() {
    test_activation::<f32>(ActivationType::SILU, 1.7, true);
}

fn test_silu_reference<T: DataTypeRepr + Float>() {
    // Include the negative tail, which is not covered by make_data(). Compute
    // the reference independently in f64, after rounding the input to T.
    let data = (-5120..=5120).map(|i| T::from(f64::from(i) / 64.0).unwrap()).collect::<Vec<_>>();
    let (rtol, atol) = match T::data_type() {
        DataType::F16 => (0.002, 2f64.powi(-24)),
        DataType::BF16 => (0.008, 0.0),
        _ => (2e-5, 0.0),
    };
    for in_place in [false, true] {
        let input = Input {
            data: data.clone().into_boxed_slice(),
            act_type: ActivationType::SILU,
            alpha: 1.0,
            in_place,
        };
        for_each_backend!(|B| {
            let actual = get_output::<T, B>(&input);
            for (index, (&x, &actual)) in input.data.iter().zip(actual.iter()).enumerate() {
                let x = x.to_f64().unwrap();
                let expected = T::from(x / (1.0 + (-x).exp())).unwrap().to_f64().unwrap();
                let actual = actual.to_f64().unwrap();
                assert!(
                    (actual - expected).abs() <= atol + rtol * expected.abs(),
                    "SiLU {:?} {} in_place={in_place} index={index} x={x}: expected={expected:e}, actual={actual:e}",
                    T::data_type(),
                    std::any::type_name::<B>()
                );
            }
        });
    }
}

#[uzu_test]
fn test_silu_reference_f32() {
    test_silu_reference::<f32>();
}

#[uzu_test]
fn test_silu_reference_f16() {
    test_silu_reference::<f16>();
}

#[uzu_test]
fn test_silu_reference_bf16() {
    test_silu_reference::<bf16>();
}

// The final result is normal even though exp(x) alone is subnormal.
fn check_silu_extreme_tail<T: DataTypeRepr + Float>() {
    let input = Input {
        data: [-91.0, -90.5, -90.0, -89.5, -89.0, -88.5, -88.0]
            .into_iter()
            .map(|x| T::from(x).unwrap())
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        act_type: ActivationType::SILU,
        alpha: 1.0,
        in_place: false,
    };
    let rtol = if T::data_type() == DataType::BF16 {
        0.008
    } else {
        2e-5
    };
    let mut errors = Vec::new();
    for_each_backend!(|B| {
        let actual = get_output::<T, B>(&input);
        for (&x, &value) in input.data.iter().zip(&actual) {
            let x = x.to_f64().unwrap();
            let expected = T::from(x / (1.0 + (-x).exp())).unwrap().to_f64().unwrap();
            let actual = value.to_f64().unwrap();
            assert!(expected.abs() >= f64::from(f32::MIN_POSITIVE));
            if !actual.is_finite() || (actual - expected).abs() > rtol * expected.abs() {
                errors.push(format!(
                    "SiLU extreme {:?} {} x={x}: expected={expected:e}, actual={actual:e}",
                    T::data_type(),
                    std::any::type_name::<B>()
                ));
            }
        }
    });
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}
#[uzu_test]
fn test_silu_extreme_tail_f32() {
    check_silu_extreme_tail::<f32>();
}

#[uzu_test]
fn test_silu_extreme_tail_bf16() {
    check_silu_extreme_tail::<bf16>();
}

fn check_silu_alpha_reference<T: DataTypeRepr + Float>() {
    let (rtol, atol) = match T::data_type() {
        DataType::F16 => (0.002, 2f64.powi(-24)),
        DataType::BF16 => (0.008, 0.0),
        _ => (2e-5, 0.0),
    };
    for alpha in [-1.7, 0.0, 0.5, 1.0, 1.7] {
        for in_place in [false, true] {
            let input = Input {
                data: [-40.0, -30.0, -20.0, -17.0, -8.0, 0.0, 0.1, 4.0, 20.0, 40.0]
                    .into_iter()
                    .map(|x| T::from(x).unwrap())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                act_type: ActivationType::SILU,
                alpha,
                in_place,
            };
            for_each_backend!(|B| {
                let actual = get_output::<T, B>(&input);
                for (&x, &value) in input.data.iter().zip(&actual) {
                    let x = x.to_f64().unwrap();
                    let expected = T::from(x / (1.0 + (-f64::from(alpha) * x).exp())).unwrap().to_f64().unwrap();
                    let actual = value.to_f64().unwrap();
                    assert!(
                        actual.is_finite() && (actual - expected).abs() <= atol + rtol * expected.abs(),
                        "SiLU alpha {:?} {} alpha={alpha} in_place={in_place} x={x}: expected={expected:e} actual={actual:e}",
                        T::data_type(),
                        std::any::type_name::<B>()
                    );
                }
            });
        }
    }
}

#[uzu_test]
fn test_silu_alpha_reference_f32() {
    check_silu_alpha_reference::<f32>();
}
#[uzu_test]
fn test_silu_alpha_reference_f16() {
    check_silu_alpha_reference::<f16>();
}
#[uzu_test]
fn test_silu_alpha_reference_bf16() {
    check_silu_alpha_reference::<bf16>();
}
