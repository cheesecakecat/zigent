const std = @import("std");
const testing = std.testing;

/// Core ECS types and functionality
pub const core = @import("ecs/core.zig");
pub const world = @import("ecs/world.zig");
pub const World = world.World;

/// Core types
pub const Entity = core.Entity;
pub const EntityId = core.EntityId;
pub const ComponentId = core.ComponentId;
pub const ComponentStorage = core.ComponentStorage;
pub const SparseSet = core.SparseSet;

test {
    // Test this file's declarations
    testing.refAllDecls(@This());

    // Test all imported modules
    _ = @import("ecs/core.zig");
    _ = @import("ecs/world.zig");
    _ = @import("ecs/tests.zig");
}
