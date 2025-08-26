use crate::util::AsBytes;
use ash::vk::SpecializationMapEntry;

// todo: potentially simplify away from builder to just have fields and progressively build?
#[derive(Clone)]
pub(crate) struct SpecConstantList {
    values: Vec<u8>,
    sizes: Vec<usize>
}

impl SpecConstantList {
    pub(crate) fn new() -> Self {
        Self {
            values: vec![],
            sizes: vec![],
        }
    }

    pub(crate) fn append<T: AsBytes>(mut self, val: T) -> Self {
        let bytes = unsafe { val.as_bytes() };

        self.values.extend_from_slice(bytes);
        self.sizes.push(size_of::<T>());

        self
    }

    pub(crate) fn build(self) -> (Vec<SpecializationMapEntry>, Vec<u8>) {
        let mut entries = vec![];
        let mut offset = 0;
        for (idx, size) in self.sizes.iter().enumerate() {
            entries.push(
                SpecializationMapEntry::default()
                    .constant_id(idx as u32)
                    .offset(offset as u32)
                    .size(*size)
            );

            offset += *size;
        }

        (entries, self.values)
    }
}