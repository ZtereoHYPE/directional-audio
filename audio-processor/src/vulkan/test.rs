// use ash::Device;
// 
// struct Parent<'a> {
//     device: Device,
//     child: ChildStruct<'a>,
// }
// 
// struct ChildStruct<'a> {
//     device: &'a Device
// }
// 
// impl<'a> Parent<'a> {
//     fn new(device: Device) -> Parent<'a> {
//         let child = ChildStruct::new(&device);
// 
//         Parent {
//             device,
//             child
//         }
//     }
// }
// 
// impl<'a> ChildStruct<'a> {
//     fn new(device: &'a Device) -> ChildStruct<'a> {
//         ChildStruct {
//             device
//         }
//     }
// }