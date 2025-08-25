use ndarray::{Array, Array2, Axis}; 
use polars::prelude::*;
use polars::prelude::ParquetReader;
use dendritic::optimizer::model::*; 


fn print_type_of<T>(_: &T) {
    println!("Type: {}", std::any::type_name::<T>());
}


pub struct IrisFlowersModel {

    /// Name of model
    name: String,

    /// Path of where dataset lives
    file_path: String,

    /// Training dataset as ndarray
    x: Array2<f64>,

    /// Target values as ndaarray
    y: Array2<f64>

}

impl ModelPipeline for IrisFlowersModel {
 
    fn register(name: &str) -> Self {
        IrisFlowersModel {
            name: name.to_string(),
            file_path: "data/iris.parquet".into(),
            x: Array2::zeros((150, 4)),
            y: Array2::zeros((150, 1))  
        }
    }

}

impl Load for IrisFlowersModel {

    fn load(&self) {

        let mut x: Array2<f64> = Array2::zeros((150, 4));
        let mut y: Array2<f64> = Array2::zeros((150, 1));

        let mut file = std::fs::File::open(&self.file_path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();
        let df_head = df.head(Some(5));

        let cols = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"];
        for (i, col_name) in cols.iter().enumerate() {

            let col_df = df.column(col_name).unwrap().f64().unwrap();
            let col_df_vec: Vec<f64> = col_df.into_no_null_iter().collect();

            let col = Array::from_vec(col_df_vec);
            x.index_axis_mut(Axis(1), i).assign(&col);
        }

        let y_col_df = df.column("species_code").unwrap().f64().unwrap();
        let y_col_df_vec: Vec<f64> = y_col_df.into_no_null_iter().collect();

        let col = Array::from_vec(y_col_df_vec);
        y.index_axis_mut(Axis(1), 0).assign(&col);

    }

}