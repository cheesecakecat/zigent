const std = @import("std");
const core = @import("core.zig");
const Allocator = std.mem.Allocator;

pub const World = struct {
    allocator: Allocator,
    entities: std.ArrayList(core.Entity),
    free_entities: std.ArrayList(core.EntityId),
    generation: u32,
    components: std.AutoHashMap(core.ComponentId, core.ComponentStorage),
    entity_components: std.AutoHashMap(core.EntityId, core.SparseSet),
    component_indices: std.AutoHashMap(core.ComponentId, usize),

    pub fn init(allocator: Allocator) !World {
        return .{
            .allocator = allocator,
            .entities = std.ArrayList(core.Entity).init(allocator),
            .free_entities = std.ArrayList(core.EntityId).init(allocator),
            .generation = 0,
            .components = std.AutoHashMap(core.ComponentId, core.ComponentStorage).init(allocator),
            .entity_components = std.AutoHashMap(core.EntityId, core.SparseSet).init(allocator),
            .component_indices = std.AutoHashMap(core.ComponentId, usize).init(allocator),
        };
    }

    pub fn deinit(self: *World) void {
        var components_iter = self.components.valueIterator();
        while (components_iter.next()) |component_storage| {
            component_storage.deinit();
        }
        self.components.deinit();

        var entity_components_iter = self.entity_components.valueIterator();
        while (entity_components_iter.next()) |sparse_set| {
            sparse_set.deinit();
        }
        self.entity_components.deinit();

        self.component_indices.deinit();
        self.entities.deinit();
        self.free_entities.deinit();
    }

    pub fn createEntity(self: *World) !core.Entity {
        const entity_id = if (self.free_entities.popOrNull()) |id|
            id
        else
            @as(core.EntityId, @intCast(self.entities.items.len));
        const entity = core.Entity.init(entity_id, self.generation);
        try self.entities.append(entity);
        return entity;
    }

    pub fn destroyEntity(self: *World, entity: core.Entity) !void {
        if (entity.id >= self.entities.items.len) return;
        if (self.entities.items[entity.id].generation != entity.generation) return;

        try self.free_entities.append(entity.id);
        self.generation += 1;
        self.entities.items[entity.id].generation = self.generation;

        if (self.entity_components.getPtr(entity.id)) |sparse_set| {
            sparse_set.deinit();
            _ = self.entity_components.remove(entity.id);
        }
    }

    pub fn addComponent(self: *World, entity: core.Entity, component_id: core.ComponentId, component: anytype) !void {
        if (entity.id >= self.entities.items.len) return error.InvalidEntity;
        if (self.entities.items[entity.id].generation != entity.generation) return error.InvalidEntity;

        var component_storage = if (self.components.getPtr(component_id)) |storage|
            storage
        else blk: {
            const storage = try core.ComponentStorage.init(self.allocator, @sizeOf(@TypeOf(component)), 8);
            try self.components.put(component_id, storage);
            break :blk self.components.getPtr(component_id).?;
        };

        const component_index = try component_storage.add(component);
        try self.component_indices.put(component_id, component_index);

        var entity_components = if (self.entity_components.getPtr(entity.id)) |sparse_set|
            sparse_set
        else blk: {
            const sparse_set = try core.SparseSet.init(self.allocator, 8);
            try self.entity_components.put(entity.id, sparse_set);
            break :blk self.entity_components.getPtr(entity.id).?;
        };

        try entity_components.add(component_id);
    }

    pub fn getComponent(self: *World, entity: core.Entity, component_id: core.ComponentId, comptime T: type) ?*T {
        if (entity.id >= self.entities.items.len) return null;
        if (self.entities.items[entity.id].generation != entity.generation) return null;

        const component_storage = self.components.getPtr(component_id) orelse return null;
        const entity_components = self.entity_components.get(entity.id) orelse return null;
        const component_index = self.component_indices.get(component_id) orelse return null;

        if (!entity_components.contains(component_id)) return null;
        return component_storage.get(T, component_index);
    }

    pub fn removeComponent(self: *World, entity: core.Entity, component_id: core.ComponentId) void {
        if (entity.id >= self.entities.items.len) return;
        if (self.entities.items[entity.id].generation != entity.generation) return;

        if (self.entity_components.getPtr(entity.id)) |entity_components| {
            entity_components.remove(component_id);
        }
    }
};
