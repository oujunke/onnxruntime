// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// scalar zero & scale with uint8
TEST(DequantizeLinearOpTest, DequantizeLinear_0) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("x", dims, {0, 3, 128, 255});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<uint8_t>("x_zero_point", {}, {128});
  test.AddOutput<float>("y", dims, {-256.0f, -250.0f, 0.0f, 254.0f});
  test.Run();
}

// scalar zero & scale with int8
TEST(DequantizeLinearOpTest, DequantizeLinear_1) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {-10});
  test.AddOutput<float>("y", dims, {-40.0f, 14.0f, 220.0f, 274.0f});
  test.Run();
}

// 1d zero & scale with uint8 broadcast axis 0
TEST(DequantizeLinearOpTest, DequantizeLinear_2) {
  OpTester test("DequantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<uint8_t>("X", dims,
                         {0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 10, 20, 30});
  test.AddInput<float>("scale", {}, {1.0f});

  test.AddInput<uint8_t>("zero_point", {}, {0});

  test.AddOutput<float>("Y", dims,
                        {0, 1, 2, 3,
                         0, 1, 2, 3,
                         0, 10, 20, 30});
  test.Run();
}

// quantize with scalar zero point and scale
TEST(QuantizeLinearOpTest, QuantizeLinear_0) {
  OpTester test("QuantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{6};
  test.AddInput<float>("x", dims, {0, 2, 3, 1000, -254, -1000});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {128});
  test.AddOutput<uint8_t>("y", dims, {128, 129, 130, 255, 1, 0});
  test.Run();
}

// quantize with broadcasting and negative axis (-2 resolves to axis 0)
TEST(QuantizeLinearOpTest, QuantizeLinear_2) {
  OpTester test("QuantizeLinear", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<float>("X", dims,
                       {0, 2, 3, 1000,
                        0, 2, 3, 1000,
                        0, 2, 3, 1000});
  test.AddInput<float>("scale", {}, {4});
  test.AddInput<uint8_t>("zero_point", {}, {0});
  test.AddOutput<uint8_t>("Y", dims,
                          {0, 1, 1, 250,
                           0, 1, 1, 250,
                           0, 1, 1, 250});
  test.Run();
}

TEST(ConvIntegerTest, ConvIntegerTest) {
  OpTester test("ConvInteger", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> x_dims{1, 1, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
						  8, 9, 10});
  std::vector<int64_t> w_dims{1, 1, 2, 2};
  test.AddInput<uint8_t>("w", w_dims,
                         {1, 1,
					      1, 1});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  std::vector<int64_t> y_dims{1, 1, 2, 2};
  test.AddOutput<int32_t>("y", y_dims,
                          {12, 16,
						   24, 28});
  test.Run();
}

TEST(ConvIntegerTest_with_padding, ConvIntegerTest) {
  OpTester test("ConvInteger", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> x_dims{1, 1, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
                          8, 9, 10});
  std::vector<int64_t> w_dims{1, 1, 2, 2};
  test.AddInput<uint8_t>("w", w_dims,
                         {1, 1,
                          1, 1});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddAttribute<std::vector<int64_t>>("pads", {1, 1, 1, 1});
  std::vector<int64_t> y_dims{1, 1, 4, 4};
  test.AddOutput<int32_t>("y", y_dims,
                          {1, 3, 5, 3,
	                       5, 12, 16, 9,
                           11, 24, 28, 15,
	                       7, 15, 17, 9});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
