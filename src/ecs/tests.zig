const std = @import("std");
const testing = std.testing;
const core = @import("core.zig");
const World = @import("world.zig").World;

const Position = struct {
    x: f32,
    y: f32,
};

const Velocity = struct {
    x: f32,
    y: f32,
};

test "entity creation and destruction" {
    // Arrange
    var world = try World.init(testing.allocator);
    defer world.deinit();

    // Act
    const entity1 = try world.createEntity();
    const entity2 = try world.createEntity();
    try world.destroyEntity(entity1);
    const entity3 = try world.createEntity();

    // Assert
    try testing.expect(entity1.id == 0);
    try testing.expect(entity2.id == 1);
    try testing.expect(entity3.id == 0);
    try testing.expect(entity3.generation > entity1.generation);
}

test "component addition and retrieval" {
    // Arrange
    var world = try World.init(testing.allocator);
    defer world.deinit();
    const entity = try world.createEntity();
    const pos = Position{ .x = 1.0, .y = 2.0 };
    const vel = Velocity{ .x = 3.0, .y = 4.0 };

    // Act
    try world.addComponent(entity, 0, pos);
    try world.addComponent(entity, 1, vel);

    // Assert
    const retrieved_pos = world.getComponent(entity, 0, Position) orelse unreachable;
    const retrieved_vel = world.getComponent(entity, 1, Velocity) orelse unreachable;
    try testing.expectEqual(pos, retrieved_pos.*);
    try testing.expectEqual(vel, retrieved_vel.*);
}

test "component removal" {
    // Arrange
    var world = try World.init(testing.allocator);
    defer world.deinit();
    const entity = try world.createEntity();
    const pos = Position{ .x = 1.0, .y = 2.0 };
    try world.addComponent(entity, 0, pos);

    // Act
    world.removeComponent(entity, 0);

    // Assert
    try testing.expect(world.getComponent(entity, 0, Position) == null);
}

test "entity reuse" {
    // Arrange
    var world = try World.init(testing.allocator);
    defer world.deinit();
    const entity1 = try world.createEntity();
    const pos1 = Position{ .x = 1.0, .y = 2.0 };
    try world.addComponent(entity1, 0, pos1);

    // Act
    try world.destroyEntity(entity1);
    const entity2 = try world.createEntity();
    const pos2 = Position{ .x = 3.0, .y = 4.0 };
    try world.addComponent(entity2, 0, pos2);

    // Assert
    try testing.expect(entity1.id == entity2.id);
    try testing.expect(entity1.generation != entity2.generation);
    const retrieved_pos = world.getComponent(entity2, 0, Position) orelse unreachable;
    try testing.expectEqual(pos2, retrieved_pos.*);
}

test "sparse set operations" {
    // Arrange
    var sparse_set = try core.SparseSet.init(testing.allocator, 8);
    defer sparse_set.deinit();

    // Act
    try sparse_set.add(5);
    try sparse_set.add(10);
    try sparse_set.add(15);
    sparse_set.remove(10);

    // Assert
    try testing.expect(sparse_set.contains(5));
    try testing.expect(!sparse_set.contains(10));
    try testing.expect(sparse_set.contains(15));
    try testing.expect(sparse_set.len == 2);
}

test "component storage resizing" {
    // Arrange
    var storage = try core.ComponentStorage.init(testing.allocator, @sizeOf(Position), 2);
    defer storage.deinit();
    const pos1 = Position{ .x = 1.0, .y = 2.0 };
    const pos2 = Position{ .x = 3.0, .y = 4.0 };
    const pos3 = Position{ .x = 5.0, .y = 6.0 };

    // Act
    const idx1 = try storage.add(pos1);
    const idx2 = try storage.add(pos2);
    const idx3 = try storage.add(pos3);

    // Assert
    try testing.expectEqual(pos1, storage.get(Position, idx1).?.*);
    try testing.expectEqual(pos2, storage.get(Position, idx2).?.*);
    try testing.expectEqual(pos3, storage.get(Position, idx3).?.*);
}
