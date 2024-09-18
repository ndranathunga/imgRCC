extern "C" {
    fn say_hello();
}

pub fn call_cpp_function() {
    unsafe {
        say_hello(); // Call the C++ function
    }
}