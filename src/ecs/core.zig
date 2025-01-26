const std = @import("std");
const Allocator = std.mem.Allocator;

pub const EntityId = u32;
pub const ComponentId = u32;

pub const Entity = struct {
    id: EntityId,
    generation: u32,

    pub fn init(id: EntityId, generation: u32) Entity {
        return .{
            .id = id,
            .generation = generation,
        };
    }
};

pub const ComponentStorage = struct {
    data: []align(8) u8,
    len: usize,
    component_size: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, component_size: usize, initial_capacity: usize) !ComponentStorage {
        const data = try allocator.alignedAlloc(u8, 8, component_size * initial_capacity);
        return .{
            .data = data,
            .len = 0,
            .component_size = component_size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ComponentStorage) void {
        self.allocator.free(self.data);
    }

    pub fn add(self: *ComponentStorage, component: anytype) !usize {
        if (self.len * self.component_size >= self.data.len) {
            const new_capacity = if (self.data.len == 0) 8 else self.data.len * 2;
            const new_data = try self.allocator.realloc(self.data, new_capacity);
            self.data = new_data;
        }

        const index = self.len;
        const raw_ptr = &self.data[index * self.component_size];
        const ptr: *@TypeOf(component) = @ptrCast(@alignCast(raw_ptr));
        ptr.* = component;
        self.len += 1;
        return index;
    }

    pub fn get(self: *ComponentStorage, comptime T: type, index: usize) ?*T {
        if (index >= self.len) return null;
        const raw_ptr = &self.data[index * self.component_size];
        const ptr: *T = @ptrCast(@alignCast(raw_ptr));
        return ptr;
    }
};

pub const SparseSet = struct {
    dense: []EntityId,
    sparse: []usize,
    len: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, initial_capacity: usize) !SparseSet {
        const dense = try allocator.alloc(EntityId, initial_capacity);
        const sparse = try allocator.alloc(usize, initial_capacity);
        return .{
            .dense = dense,
            .sparse = sparse,
            .len = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SparseSet) void {
        self.allocator.free(self.dense);
        self.allocator.free(self.sparse);
    }

    pub fn add(self: *SparseSet, entity: EntityId) !void {
        if (entity >= self.sparse.len) {
            const new_capacity = @max(entity + 1, self.sparse.len * 2);
            self.sparse = try self.allocator.realloc(self.sparse, new_capacity);
        }

        if (self.len >= self.dense.len) {
            const new_capacity = if (self.dense.len == 0) 8 else self.dense.len * 2;
            self.dense = try self.allocator.realloc(self.dense, new_capacity);
        }

        self.sparse[entity] = self.len;
        self.dense[self.len] = entity;
        self.len += 1;
    }

    pub fn contains(self: *const SparseSet, entity: EntityId) bool {
        return entity < self.sparse.len and
            self.sparse[entity] < self.len and
            self.dense[self.sparse[entity]] == entity;
    }

    pub fn remove(self: *SparseSet, entity: EntityId) void {
        if (!self.contains(entity)) return;

        const index = self.sparse[entity];
        const last = self.len - 1;
        const last_entity = self.dense[last];

        self.dense[index] = last_entity;
        self.sparse[last_entity] = index;
        self.len -= 1;
    }
};
