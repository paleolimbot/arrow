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

#pragma once

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>

#include "arrow/result.h"
#include "arrow/util/endian.h"
#include "arrow/util/macros.h"
#include "arrow/util/ubsan.h"

#include "parquet/platform.h"

namespace parquet::geometry {

constexpr double kInf = std::numeric_limits<double>::infinity();

enum class Dimensions { XY = 0, XYZ = 1, XYM = 2, XYZM = 3, MIN = 0, MAX = 3 };

enum class GeometryType {
  POINT = 1,
  LINESTRING = 2,
  POLYGON = 3,
  MULTIPOINT = 4,
  MULTILINESTRING = 5,
  MULTIPOLYGON = 6,
  GEOMETRYCOLLECTION = 7,
  MIN = 1,
  MAX = 7
};

struct BoundingBox {
  using XY = std::array<double, 2>;
  using XYZ = std::array<double, 3>;
  using XYM = std::array<double, 3>;
  using XYZM = std::array<double, 4>;

  BoundingBox(const XYZM& mins, const XYZM& maxes) : min(mins), max(maxes) {}
  BoundingBox() : min{kInf, kInf, kInf, kInf}, max{-kInf, -kInf, -kInf, -kInf} {}

  BoundingBox(const BoundingBox& other) = default;
  BoundingBox& operator=(const BoundingBox&) = default;

  void UpdateXY(const XY& coord) { UpdateInternal(coord); }

  void UpdateXYZ(const XYZ& coord) { UpdateInternal(coord); }

  void UpdateXYM(const XYM& coord) {
    min[0] = std::min(min[0], coord[0]);
    min[1] = std::min(min[1], coord[1]);
    min[3] = std::min(min[3], coord[2]);
    max[0] = std::max(max[0], coord[0]);
    max[1] = std::max(max[1], coord[1]);
    max[3] = std::max(max[3], coord[2]);
  }

  void UpdateXYZM(const XYZM& coord) { UpdateInternal(coord); }

  void Reset() {
    for (int i = 0; i < 4; i++) {
      min[i] = kInf;
      max[i] = -kInf;
    }
  }

  void Merge(const BoundingBox& other) {
    for (int i = 0; i < 4; i++) {
      min[i] = std::min(min[i], other.min[i]);
      max[i] = std::max(max[i], other.max[i]);
    }
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "BoundingBox [" << min[0] << " => " << max[0];
    for (int i = 1; i < 4; i++) {
      ss << ", " << min[i] << " => " << max[i];
    }

    ss << "]";

    return ss.str();
  }

  XYZM min;
  XYZM max;

 private:
  // This works for XY, XYZ, and XYZM
  template <typename Coord>
  void UpdateInternal(Coord coord) {
    static_assert(coord.size() <= 4);

    for (size_t i = 0; i < coord.size(); i++) {
      min[i] = std::min(min[i], coord[i]);
      max[i] = std::max(max[i], coord[i]);
    }
  }
};

inline bool operator==(const BoundingBox& lhs, const BoundingBox& rhs) {
  return lhs.min == rhs.min && lhs.max == rhs.max;
}

class WKBBuffer;

class PARQUET_EXPORT WKBGeometryBounder {
 public:
  WKBGeometryBounder() = default;
  WKBGeometryBounder(const WKBGeometryBounder&) = default;

  ::arrow::Status ReadGeometry(WKBBuffer* src);

  void ReadBox(const BoundingBox& box) { box_.Merge(box); }

  void ReadGeometryTypes(const std::vector<int32_t>& geospatial_types) {
    geospatial_types_.insert(geospatial_types.begin(), geospatial_types.end());
  }

  const BoundingBox& Bounds() const { return box_; }

  std::vector<int32_t> GeometryTypes() const {
    std::vector<int32_t> out(geospatial_types_.begin(), geospatial_types_.end());
    std::sort(out.begin(), out.end());
    return out;
  }

  void Reset() {
    box_.Reset();
    geospatial_types_.clear();
  }

 private:
  BoundingBox box_;
  std::unordered_set<int32_t> geospatial_types_;

  ::arrow::Status ReadGeometryInternal(WKBBuffer* src, bool record_wkb_type);

  ::arrow::Status ReadSequence(WKBBuffer* src, Dimensions dimensions, uint32_t n_coords,
                               bool swap);
};

class WKBBuffer {
 public:
  WKBBuffer() : data_(NULLPTR), size_(0) {}
  WKBBuffer(const uint8_t* data, int64_t size) : data_(data), size_(size) {}

  void Init(const uint8_t* data, int64_t size) {
    data_ = data;
    size_ = size;
  }

  ::arrow::Result<uint8_t> ReadUInt8() { return ReadChecked<uint8_t>(); }

  ::arrow::Result<uint32_t> ReadUInt32(bool swap) {
    ARROW_ASSIGN_OR_RAISE(auto value, ReadChecked<uint32_t>());
    if (ARROW_PREDICT_FALSE(swap)) {
      return value = ByteSwap(value);
    } else {
      return value;
    }
  }

  template <typename Coord, typename Visit>
  ::arrow::Status ReadDoubles(uint32_t n_coords, bool swap, Visit&& visit) {
    size_t total_bytes = n_coords * sizeof(Coord);
    if (size_ < total_bytes) {
      return ::arrow::Status::SerializationError(
          "Can't coordinate sequence of ", total_bytes, " bytes from WKBBuffer with ",
          size_, " remaining");
    }

    if (ARROW_PREDICT_FALSE(swap)) {
      Coord coord;
      for (uint32_t i = 0; i < n_coords; i++) {
        coord = ReadUnchecked<Coord>();
        for (uint32_t j = 0; j < coord.size(); j++) {
          coord[j] = ByteSwap(coord[j]);
        }

        visit(coord);
      }
    } else {
      for (uint32_t i = 0; i < n_coords; i++) {
        visit(ReadUnchecked<Coord>());
      }
    }

    return ::arrow::Status::OK();
  }

  size_t size() { return size_; }

 private:
  const uint8_t* data_;
  size_t size_;

  template <typename T>
  ::arrow::Result<T> ReadChecked() {
    if (size_ < sizeof(T)) {
      return ::arrow::Status::SerializationError(
          "Can't read ", sizeof(T), " bytes from WKBBuffer with ", size_, " remaining");
    }

    return ReadUnchecked<T>();
  }

  template <typename T>
  T ReadUnchecked() {
    T out = ::arrow::util::SafeLoadAs<T>(data_);
    data_ += sizeof(T);
    size_ -= sizeof(T);
    return out;
  }

  template <typename T>
  T ByteSwap(T value) {
    return ::arrow::bit_util::ByteSwap(value);
  }
};

}  // namespace parquet::geometry
