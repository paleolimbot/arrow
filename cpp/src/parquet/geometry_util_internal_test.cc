// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

#include "arrow/testing/gtest_compat.h"

#include "parquet/geometry_util_internal.h"

namespace parquet::geometry {

TEST(TestGeometryUtil, TestDimensions) {
  EXPECT_EQ(Dimensions::size(Dimensions::dimensions::XY), 2);
  EXPECT_EQ(Dimensions::size(Dimensions::dimensions::XYZ), 3);
  EXPECT_EQ(Dimensions::size(Dimensions::dimensions::XYM), 3);
  EXPECT_EQ(Dimensions::size(Dimensions::dimensions::XYZM), 4);

  EXPECT_EQ(Dimensions::ToString(Dimensions::dimensions::XY), "XY");
  EXPECT_EQ(Dimensions::ToString(Dimensions::dimensions::XYZ), "XYZ");
  EXPECT_EQ(Dimensions::ToString(Dimensions::dimensions::XYM), "XYM");
  EXPECT_EQ(Dimensions::ToString(Dimensions::dimensions::XYZM), "XYZM");

  EXPECT_EQ(Dimensions::FromWKB(1), Dimensions::dimensions::XY);
  EXPECT_EQ(Dimensions::FromWKB(1001), Dimensions::dimensions::XYZ);
  EXPECT_EQ(Dimensions::FromWKB(2001), Dimensions::dimensions::XYM);
  EXPECT_EQ(Dimensions::FromWKB(3001), Dimensions::dimensions::XYZM);
  EXPECT_THROW(Dimensions::FromWKB(4001), ParquetException);
}

TEST(TestGeometryUtil, TestGeometryType) {
  EXPECT_EQ(GeometryType::ToString(GeometryType::geometry_type::POINT), "POINT");
  EXPECT_EQ(GeometryType::ToString(GeometryType::geometry_type::LINESTRING),
            "LINESTRING");
  EXPECT_EQ(GeometryType::ToString(GeometryType::geometry_type::POLYGON), "POLYGON");
  EXPECT_EQ(GeometryType::ToString(GeometryType::geometry_type::MULTIPOINT),
            "MULTIPOINT");
  EXPECT_EQ(GeometryType::ToString(GeometryType::geometry_type::MULTILINESTRING),
            "MULTILINESTRING");
  EXPECT_EQ(GeometryType::ToString(GeometryType::geometry_type::MULTIPOLYGON),
            "MULTIPOLYGON");
  EXPECT_EQ(GeometryType::ToString(GeometryType::geometry_type::GEOMETRYCOLLECTION),
            "GEOMETRYCOLLECTION");

  EXPECT_EQ(GeometryType::FromWKB(1), GeometryType::geometry_type::POINT);
  EXPECT_EQ(GeometryType::FromWKB(1001), GeometryType::geometry_type::POINT);
  EXPECT_EQ(GeometryType::FromWKB(1002), GeometryType::geometry_type::LINESTRING);
  EXPECT_EQ(GeometryType::FromWKB(1003), GeometryType::geometry_type::POLYGON);
  EXPECT_EQ(GeometryType::FromWKB(1004), GeometryType::geometry_type::MULTIPOINT);
  EXPECT_EQ(GeometryType::FromWKB(1005), GeometryType::geometry_type::MULTILINESTRING);
  EXPECT_EQ(GeometryType::FromWKB(1006), GeometryType::geometry_type::MULTIPOLYGON);
  EXPECT_EQ(GeometryType::FromWKB(1007), GeometryType::geometry_type::GEOMETRYCOLLECTION);
  EXPECT_THROW(GeometryType::FromWKB(1100), ParquetException);
}

TEST(TestGeometryUtil, TestBoundingBox) {
  BoundingBox box;
  EXPECT_EQ(box, BoundingBox(Dimensions::dimensions::XYZM, {kInf, kInf, kInf, kInf},
                             {-kInf, -kInf, -kInf, -kInf}));
  EXPECT_EQ(box.ToString(),
            "BoundingBox XYZM [inf => -inf, inf => -inf, inf => -inf, inf => -inf]");

  BoundingBox box_xyzm(Dimensions::dimensions::XYZM, {-1, -2, -3, -4}, {1, 2, 3, 4});

  BoundingBox box_xy(Dimensions::dimensions::XY, {-10, -20, kInf, kInf},
                     {10, 20, -kInf, -kInf});
  BoundingBox box_xyz(Dimensions::dimensions::XYZ, {kInf, kInf, -30, kInf},
                      {-kInf, -kInf, 30, -kInf});
  BoundingBox box_xym(Dimensions::dimensions::XYM, {kInf, kInf, -40, kInf},
                      {-kInf, -kInf, 40, -kInf});

  box_xyzm.Merge(box_xy);
  EXPECT_EQ(box_xyzm, BoundingBox(Dimensions::dimensions::XYZM, {-10, -20, -3, -4},
                                  {10, 20, 3, 4}));

  box_xyzm.Merge(box_xyz);
  EXPECT_EQ(box_xyzm, BoundingBox(Dimensions::dimensions::XYZM, {-10, -20, -30, -4},
                                  {10, 20, 30, 4}));

  box_xyzm.Merge(box_xym);
  EXPECT_EQ(box_xyzm, BoundingBox(Dimensions::dimensions::XYZM, {-10, -20, -30, -40},
                                  {10, 20, 30, 40}));

  box_xyzm.Reset();
  EXPECT_EQ(box_xyzm, BoundingBox());
}

struct WKBTestCase {
  WKBTestCase() = default;
  WKBTestCase(GeometryType::geometry_type x, Dimensions::dimensions y,
              const std::vector<uint8_t>& z, const std::vector<double>& box_values = {})
      : geometry_type(x), dimensions(y), wkb(z) {
    std::array<double, 4> mins = {kInf, kInf, kInf, kInf};
    std::array<double, 4> maxes{-kInf, -kInf, -kInf, -kInf};
    for (uint32_t i = 0; i < Dimensions::size(y); i++) {
      mins[i] = box_values[i];
      maxes[i] = box_values[Dimensions::size(y) + i];
    }
    box = BoundingBox(y, mins, maxes).ToXYZM();
  }
  WKBTestCase(const WKBTestCase& other) = default;

  GeometryType::geometry_type geometry_type;
  Dimensions::dimensions dimensions;
  std::vector<uint8_t> wkb;
  BoundingBox box;
};

std::ostream& operator<<(std::ostream& os, const WKBTestCase& obj) {
  os << GeometryType::ToString(obj.geometry_type) << " "
     << Dimensions::ToString(obj.dimensions);
  return os;
}

std::ostream& operator<<(std::ostream& os, const BoundingBox& obj) {
  os << obj.ToString();
  return os;
}

class WKBTestFixture : public ::testing::TestWithParam<WKBTestCase> {
 protected:
  WKBTestCase test_case;
};

TEST_P(WKBTestFixture, TestWKBBounderNonEmpty) {
  auto item = GetParam();

  WKBGeometryBounder bounder;
  EXPECT_EQ(bounder.Bounds(), BoundingBox());

  WKBBuffer buf(item.wkb.data(), item.wkb.size());
  bounder.ReadGeometry(&buf);
  EXPECT_EQ(buf.size(), 0);

  bounder.Flush();
  EXPECT_EQ(bounder.Bounds(), item.box);
  uint32_t wkb_type =
      static_cast<int>(item.dimensions) * 1000 + static_cast<int>(item.geometry_type);
  EXPECT_THAT(bounder.GeometryTypes(), ::testing::ElementsAre(::testing::Eq(wkb_type)));

  bounder.Reset();
  EXPECT_EQ(bounder.Bounds(), BoundingBox());
  EXPECT_TRUE(bounder.GeometryTypes().empty());
}

INSTANTIATE_TEST_SUITE_P(
    TestGeometryUtil, WKBTestFixture,
    ::testing::Values(
        // POINT (30 10)
        WKBTestCase(GeometryType::geometry_type::POINT, Dimensions::dimensions::XY,
                    {0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40},
                    {30, 10, 30, 10}),
        // POINT Z (30 10 40)
        WKBTestCase(GeometryType::geometry_type::POINT, Dimensions::dimensions::XYZ,
                    {0x01, 0xe9, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24,
                     0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40},
                    {30, 10, 40, 30, 10, 40}),
        // POINT M (30 10 300)
        WKBTestCase(GeometryType::geometry_type::POINT, Dimensions::dimensions::XYM,
                    {0x01, 0xd1, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24,
                     0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40},
                    {30, 10, 300, 30, 10, 300}),
        // POINT ZM (30 10 40 300)
        WKBTestCase(GeometryType::geometry_type::POINT, Dimensions::dimensions::XYZM,
                    {0x01, 0xb9, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24,
                     0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40},
                    {30, 10, 40, 300, 30, 10, 40, 300}),
        // LINESTRING (30 10, 10 30, 40 40)
        WKBTestCase(GeometryType::geometry_type::LINESTRING, Dimensions::dimensions::XY,
                    {0x01, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e,
                     0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40},
                    {10, 10, 40, 40}),
        // LINESTRING Z (30 10 40, 10 30 40, 40 40 80)
        WKBTestCase(GeometryType::geometry_type::LINESTRING, Dimensions::dimensions::XYZ,
                    {0x01, 0xea, 0x03, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x54, 0x40},
                    {10, 10, 40, 40, 40, 80}),
        // LINESTRING M (30 10 300, 10 30 300, 40 40 1600)
        WKBTestCase(GeometryType::geometry_type::LINESTRING, Dimensions::dimensions::XYM,
                    {0x01, 0xd2, 0x07, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0,
                     0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x99, 0x40},
                    {10, 10, 300, 40, 40, 1600}),
        // LINESTRING ZM (30 10 40 300, 10 30 40 300, 40 40 80 1600)
        WKBTestCase(GeometryType::geometry_type::LINESTRING, Dimensions::dimensions::XYZM,
                    {0x01, 0xba, 0x0b, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44,
                     0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54, 0x40, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x99, 0x40},
                    {10, 10, 40, 300, 40, 40, 80, 1600}),
        // POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))
        WKBTestCase(GeometryType::geometry_type::POLYGON, Dimensions::dimensions::XY,
                    {0x01, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44,
                     0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x24, 0x40},
                    {10, 10, 40, 40}),
        // POLYGON Z ((30 10 40, 40 40 80, 20 40 60, 10 20 30, 30 10 40))
        WKBTestCase(
            GeometryType::geometry_type::POLYGON, Dimensions::dimensions::XYZ,
            {0x01, 0xeb, 0x03, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44,
             0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54, 0x40, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x4e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x44, 0x40},
            {10, 10, 30, 40, 40, 80}),
        // POLYGON M ((30 10 300, 40 40 1600, 20 40 800, 10 20 200, 30 10 300))
        WKBTestCase(
            GeometryType::geometry_type::POLYGON, Dimensions::dimensions::XYM,
            {0x01, 0xd3, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44,
             0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x99, 0x40, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x89, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x69, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
             0xc0, 0x72, 0x40},
            {10, 10, 200, 40, 40, 1600}),
        // POLYGON ZM ((30 10 40 300, 40 40 80 1600, 20 40 60 800, 10 20 30 200, 30 10 40
        // 300))
        WKBTestCase(
            GeometryType::geometry_type::POLYGON, Dimensions::dimensions::XYZM,
            {0x01, 0xbb, 0x0b, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00,
             0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44,
             0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x54, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x99, 0x40, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4e, 0x40, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x89, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x69, 0x40, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24,
             0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
             0x00, 0xc0, 0x72, 0x40},
            {10, 10, 30, 200, 40, 40, 80, 1600}),
        // MULTIPOINT ((30 10))
        WKBTestCase(GeometryType::geometry_type::MULTIPOINT, Dimensions::dimensions::XY,
                    {0x01, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                     0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40},
                    {30, 10, 30, 10}),
        // MULTIPOINT Z ((30 10 40))
        WKBTestCase(GeometryType::geometry_type::MULTIPOINT, Dimensions::dimensions::XYZ,
                    {0x01, 0xec, 0x03, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                     0xe9, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40},
                    {30, 10, 40, 30, 10, 40}),
        // MULTIPOINT M ((30 10 300))
        WKBTestCase(GeometryType::geometry_type::MULTIPOINT, Dimensions::dimensions::XYM,
                    {0x01, 0xd4, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                     0xd1, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40},
                    {30, 10, 300, 30, 10, 300}),
        // MULTIPOINT ZM ((30 10 40 300))
        WKBTestCase(GeometryType::geometry_type::MULTIPOINT, Dimensions::dimensions::XYZM,
                    {0x01, 0xbc, 0x0b, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                     0xb9, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0xc0, 0x72, 0x40},
                    {30, 10, 40, 300, 30, 10, 40, 300}),
        // MULTILINESTRING ((30 10, 10 30, 40 40))
        WKBTestCase(GeometryType::geometry_type::MULTILINESTRING,
                    Dimensions::dimensions::XY,
                    {0x01, 0x05, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x02,
                     0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24,
                     0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40},
                    {10, 10, 40, 40}),
        // MULTILINESTRING Z ((30 10 40, 10 30 40, 40 40 80))
        WKBTestCase(
            GeometryType::geometry_type::MULTILINESTRING, Dimensions::dimensions::XYZ,
            {0x01, 0xed, 0x03, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xea, 0x03, 0x00,
             0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44,
             0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54, 0x40},
            {10, 10, 40, 40, 40, 80}),
        // MULTILINESTRING M ((30 10 300, 10 30 300, 40 40 1600))
        WKBTestCase(
            GeometryType::geometry_type::MULTILINESTRING, Dimensions::dimensions::XYM,
            {0x01, 0xd5, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xd2, 0x07, 0x00,
             0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
             0xc0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72,
             0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x99, 0x40},
            {10, 10, 300, 40, 40, 1600}),
        // MULTILINESTRING ZM ((30 10 40 300, 10 30 40 300, 40 40 80 1600))
        WKBTestCase(
            GeometryType::geometry_type::MULTILINESTRING, Dimensions::dimensions::XYZM,
            {0x01, 0xbd, 0x0b, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xba, 0x0b, 0x00,
             0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e,
             0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
             0x00, 0xc0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x54, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x99, 0x40},
            {10, 10, 40, 300, 40, 40, 80, 1600}),
        // MULTIPOLYGON (((30 10, 40 40, 20 40, 10 20, 30 10)))
        WKBTestCase(
            GeometryType::geometry_type::MULTIPOLYGON, Dimensions::dimensions::XY,
            {0x01, 0x06, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x03, 0x00, 0x00,
             0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40},
            {10, 10, 40, 40}),
        // MULTIPOLYGON Z (((30 10 40, 40 40 80, 20 40 60, 10 20 30, 30 10 40)))
        WKBTestCase(
            GeometryType::geometry_type::MULTIPOLYGON, Dimensions::dimensions::XYZ,
            {0x01, 0xee, 0x03, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xeb, 0x03, 0x00,
             0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x54, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x4e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e,
             0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40},
            {10, 10, 30, 40, 40, 80}),
        // MULTIPOLYGON M (((30 10 300, 40 40 1600, 20 40 800, 10 20 200, 30 10 300)))
        WKBTestCase(
            GeometryType::geometry_type::MULTIPOLYGON, Dimensions::dimensions::XYM,
            {0x01, 0xd6, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xd3, 0x07, 0x00,
             0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00,
             0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x99, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40,
             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x89, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00,
             0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x69,
             0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00,
             0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40},
            {10, 10, 200, 40, 40, 1600}),
        // MULTIPOLYGON ZM (((30 10 40 300, 40 40 80 1600, 20 40 60 800, 10 20 30 200, 30
        // 10 40 300)))
        WKBTestCase(GeometryType::geometry_type::MULTIPOLYGON,
                    Dimensions::dimensions::XYZM,
                    {0x01, 0xbe, 0x0b, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xbb,
                     0x0b, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54,
                     0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x99, 0x40, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4e, 0x40,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x89, 0x40, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x69, 0x40, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24,
                     0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0xc0, 0x72, 0x40},
                    {10, 10, 30, 200, 40, 40, 80, 1600}),
        // GEOMETRYCOLLECTION (POINT (30 10))
        WKBTestCase(GeometryType::geometry_type::GEOMETRYCOLLECTION,
                    Dimensions::dimensions::XY,
                    {0x01, 0x07, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                     0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40},
                    {30, 10, 30, 10}),
        // GEOMETRYCOLLECTION Z (POINT Z (30 10 40))
        WKBTestCase(GeometryType::geometry_type::GEOMETRYCOLLECTION,
                    Dimensions::dimensions::XYZ,
                    {0x01, 0xef, 0x03, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                     0xe9, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40},
                    {30, 10, 40, 30, 10, 40}),
        // GEOMETRYCOLLECTION M (POINT M (30 10 300))
        WKBTestCase(GeometryType::geometry_type::GEOMETRYCOLLECTION,
                    Dimensions::dimensions::XYM,
                    {0x01, 0xd7, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                     0xd1, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40},
                    {30, 10, 300, 30, 10, 300}),
        // GEOMETRYCOLLECTION ZM (POINT ZM (30 10 40 300))
        WKBTestCase(GeometryType::geometry_type::GEOMETRYCOLLECTION,
                    Dimensions::dimensions::XYZM,
                    {0x01, 0xbf, 0x0b, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                     0xb9, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0xc0, 0x72, 0x40},
                    {30, 10, 40, 300, 30, 10, 40, 300})));

TEST(TestGeometryUtil, MakeCoveringWKBFromBound) {
  std::string wkb_covering = MakeCoveringWKBFromBound(10, 20, 30, 40);
  // POLYGON ((10 30, 20 30, 20 40, 10 40, 10 30))
#ifdef ARROW_LITTLE_ENDIAN
  std::vector<uint8_t> expected_wkb = {
      0x01, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e,
      0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24,
      0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40};
#else
  std::vector<uint8_t> expected_wkb = {
      0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x40,
      0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x40, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x3e, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x40, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x44, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
      0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x40, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
#endif
  EXPECT_EQ(expected_wkb.size(), wkb_covering.size());
  EXPECT_EQ(0, memcmp(wkb_covering.data(), expected_wkb.data(), expected_wkb.size()));
}

struct MakeWKBPointTestCase {
  MakeWKBPointTestCase() = default;
  MakeWKBPointTestCase(const std::vector<double> xyzm, bool has_z, bool has_m)
      : has_z(has_z), has_m(has_m) {
    memcpy(this->xyzm, xyzm.data(), sizeof(this->xyzm));
  }

  double xyzm[4];
  bool has_z;
  bool has_m;
};

class MakeWKBPointTestFixture : public testing::TestWithParam<MakeWKBPointTestCase> {};

TEST_P(MakeWKBPointTestFixture, MakeWKBPoint) {
  auto param = GetParam();
  std::string wkb = MakeWKBPoint(param.xyzm, param.has_z, param.has_m);
  WKBGeometryBounder bounder;
  WKBBuffer buf(reinterpret_cast<uint8_t*>(wkb.data()), wkb.size());
  bounder.ReadGeometry(&buf);
  bounder.Flush();
  const double* mins = bounder.Bounds().min;
  EXPECT_DOUBLE_EQ(param.xyzm[0], mins[0]);
  EXPECT_DOUBLE_EQ(param.xyzm[1], mins[1]);
  if (param.has_z) {
    EXPECT_DOUBLE_EQ(param.xyzm[2], mins[2]);
  } else {
    EXPECT_TRUE(std::isinf(mins[2]));
  }
  if (param.has_m) {
    EXPECT_DOUBLE_EQ(param.xyzm[3], mins[3]);
  } else {
    EXPECT_TRUE(std::isinf(mins[3]));
  }
}

INSTANTIATE_TEST_SUITE_P(
    TestGeometryUtil, MakeWKBPointTestFixture,
    ::testing::Values(MakeWKBPointTestCase({30, 10, 40, 300}, false, false),
                      MakeWKBPointTestCase({30, 10, 40, 300}, true, false),
                      MakeWKBPointTestCase({30, 10, 40, 300}, false, true),
                      MakeWKBPointTestCase({30, 10, 40, 300}, true, true)));

}  // namespace parquet::geometry
