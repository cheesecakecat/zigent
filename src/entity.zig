//! Zigent Entity-Component-System
//! Copyright (c) 2024 Zigent Contributors
//! SPDX-License-Identifier: MIT OR Apache-2.0

const std = @import("std");
const index = @import("entity/index.zig");
const generation = @import("entity/generation.zig");
const assert = std.debug.assert;
const math = std.math;

/// Entity represents a lightweight handle in the ECS architecture, combining a 32-bit index
/// with a 32-bit generation counter in an optimized 64-bit structure. This design provides
/// both efficient entity management and robust safety guarantees.
///
/// The generation counter prevents the "ABA problem" common in entity systems - when an entity
/// is deleted and its index recycled, the generation is incremented, invalidating any stale
/// references to the previous entity at that index.
///
/// Implementation Notes:
/// The struct maintains an exact size of 64 bits with cache-line friendly alignment.
/// Memory layout is SIMD-compatible for vectorized operations. All common operations
/// use branchless implementations for consistent performance. The index and generation
/// fields are carefully arranged to allow direct bitwise operations on the entire
/// 64-bit structure.
///
/// Performance Tips:
/// Consider entity handles as opaque identifiers and pass them by value.
/// Use batch operations when processing multiple entities. Take advantage of
/// the SIMD-compatible memory layout for vectorized operations. The hash()
/// function is optimized for use in hash maps and requires no additional computation.
///
/// Safety Notes:
/// Generation values start at 1, with 0 reserved as invalid. This helps catch
/// use-after-free bugs and uninitialized entity references. The max_index value
/// is reserved for null entities, providing a guaranteed invalid index range.
/// Always validate entities through isNull() before use.
///
/// Avoid direct struct initialization - always use provided functions.
/// Never assume consecutive indices represent valid entity ranges.
/// Remember that a matching index doesn't guarantee entity validity.
/// Don't store entity handles long-term without generation checking.
///
/// Example Usage:
/// ```
/// const entity = Entity.fromParts(1, 1);
/// if (!entity.isNull()) {
///     // Use the entity...
/// }
/// ```
pub const Entity = extern struct {
    const Self = @This();

    /// The entity's index in the pool. Aligned to 64 bits for optimal memory access
    /// and SIMD operations. Valid indices range from 0 to max_index-1, with max_index
    /// reserved for representing null entities.
    index: index.IndexType align(@alignOf(u64)),

    /// The generation counter that increments when an entity index is recycled.
    /// This provides temporal safety by invalidating stale references when an index
    /// is reused for a new entity.
    generation: generation.GenerationType,

    // Compile-time validation ensures the memory layout meets our requirements
    // for size, alignment, and field positioning.
    comptime {
        assert(@sizeOf(Self) == 8);
        assert(@alignOf(Self) == 8);
        assert(@offsetOf(Self, "index") == 0);
        assert(@offsetOf(Self, "generation") == 4);
    }

    /// Maximum valid index value, used to represent null entities. This value is
    /// placed at the highest possible index to avoid conflicts with valid entities.
    pub const max_index: index.IndexType = math.maxInt(index.IndexType);

    /// Creates a null entity value for representing invalid or uninitialized entities.
    /// The resulting entity is guaranteed to never collide with valid entity handles.
    pub inline fn initNull() Self {
        @setRuntimeSafety(false);
        return .{
            .index = max_index,
            .generation = 0,
        };
    }

    /// Efficiently checks if this entity is a null entity. This operation requires
    /// no lookups or external state checks.
    pub inline fn isNull(self: Self) bool {
        @setRuntimeSafety(false);
        return self.index == max_index and self.generation == 0;
    }

    /// Performs a fast equality comparison between two entities. Entities are equal
    /// only if both their index and generation match exactly. Uses efficient bitwise
    /// comparison of the entire 64-bit structure.
    pub inline fn eql(self: Self, other: Self) bool {
        @setRuntimeSafety(false);
        return @as(u64, @bitCast(self)) == @as(u64, @bitCast(other));
    }

    /// Generates a hash suitable for use in hash maps. The 64-bit entity structure
    /// provides good distribution properties without requiring additional computation.
    pub inline fn hash(self: Self) u64 {
        @setRuntimeSafety(false);
        return @as(u64, @bitCast(self));
    }

    /// Provides a human-readable string representation for debugging and logging.
    /// Formats null entities as "Entity(null)" and valid entities as "Entity(idx:X,gen:Y)".
    pub fn format(
        self: Self,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        if (self.isNull()) {
            try writer.writeAll("Entity(null)");
        } else {
            try writer.print("Entity(idx:{d},gen:{d})", .{
                self.index,
                self.generation,
            });
        }
    }

    /// Creates an entity from explicit index and generation values. This is a low-level
    /// function that bypasses normal entity creation flow and should be used with caution.
    /// The provided index must be less than max_index.
    pub inline fn fromParts(idx: index.IndexType, gen: generation.GenerationType) Self {
        @setRuntimeSafety(false);
        assert(idx < max_index);
        return .{
            .index = idx,
            .generation = gen,
        };
    }

    /// Decomposes an entity into its constituent parts. This is primarily useful
    /// for serialization or integration with external systems that need direct
    /// access to the index and generation values.
    pub inline fn toParts(self: Self) struct { index: index.IndexType, generation: generation.GenerationType } {
        @setRuntimeSafety(false);
        return .{
            .index = self.index,
            .generation = self.generation,
        };
    }
};

test "entity basic" {
    const testing = std.testing;

    const e1 = Entity.fromParts(1, 1);
    const e2 = Entity.fromParts(1, 2);
    const e3 = Entity.fromParts(1, 1);

    try testing.expect(e1.eql(e3));
    try testing.expect(!e1.eql(e2));
    try testing.expect(e1.hash() == e3.hash());
    try testing.expect(e1.hash() != e2.hash());

    const parts = e1.toParts();
    try testing.expectEqual(parts.index, 1);
    try testing.expectEqual(parts.generation, 1);
}

test "entity null" {
    const testing = std.testing;

    const null_entity = Entity.initNull();
    const valid_entity = Entity.fromParts(1, 1);

    try testing.expect(null_entity.isNull());
    try testing.expect(!valid_entity.isNull());
    try testing.expect(!null_entity.eql(valid_entity));
}

test "entity formatting" {
    const testing = std.testing;

    const e = Entity.fromParts(42, 7);
    const null_e = Entity.initNull();

    var buf: [32]u8 = undefined;

    try testing.expectEqualStrings(
        "Entity(idx:42,gen:7)",
        try std.fmt.bufPrint(&buf, "{}", .{e}),
    );

    try testing.expectEqualStrings(
        "Entity(null)",
        try std.fmt.bufPrint(&buf, "{}", .{null_e}),
    );
}
