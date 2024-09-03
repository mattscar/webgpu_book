// Create top-level asynchronous function
async function runExample() {

    // Obtain Response containing file content
    let resp = await fetch("simplewasm.wasm");  
    
    // Obtain the ResultSet containing the WebAssembly module and instance
    let wasm = await WebAssembly.instantiateStreaming(resp);

    // Print the return value of the exported function
    console.log(wasm.instance.exports.foo());
}

// Run example function
runExample();