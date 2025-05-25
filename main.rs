use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

fn main() {
    // unsafe {
    //     std::env::set_var("PYTHONHOME", "../.venv");
    //     std::env::set_var("PYTHONPATH", "../.venv/lib/python3.12/site-packages");
    // }

    println!("Rust: Python interpreter initialized.");
    match python_test() {
        Ok(_) => {},
        Err(e) => {
            println!("Error {}",e);
        },
    }
    
}


fn python_test() -> PyResult<()>{
    
    Python::with_gil(|py|{

    let sys = py.import("sys")?;
        // let sys = PyModule::import(py, "sys")?;
    let path = sys.getattr("path")?;
    let path = path.downcast::<PyList>()?;
    path.insert(0, ".")?; // Add current directory to the path

    println!("Rust: Added current directory to Python's sys.path.");
        let version = sys.getattr("version")?;
        let version: &str = version.extract()?;
        println!("🐍 Python version in use: {}", version);
        let executable= sys.getattr("executable")?;
        let executable: &str = executable.extract()?;
        println!("🐍 Python executable in use: {}", executable);
        let gluonts = py.import("gluonts.torch.model.simple_feedforward")?;
        // let gluonts = py.import(name)
        
        let estimator = gluonts.getattr("SimpleFeedForwardEstimator")?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("prediction_length", 24)?;
        kwargs.set_item("freq", "1H")?;
        kwargs.set_item("context_length", 72)?;
        kwargs.set_item("trainer", py.import("gluonts.trainer")?.getattr("Trainer")?.call0()?)?;

        let model = estimator.call((), Some(&kwargs))?;

        println!("Created model: {:?}", model);
        Ok(())
    })
}
