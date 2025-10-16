use ndarray::{s, Array2}; 
use polars::prelude::*;
use polars::prelude::ParquetReader;
use dendritic::optimizer::prelude::*;
use dendritic::preprocessing::prelude::*; 

pub struct HousePrices {

    /// Name of model
    name: String,

    /// Path of where dataset lives
    file_path: String,

    /// Training dataset as ndarray
    x: Array2<f64>,

    /// Target values as ndaarray
    y: Array2<f64>,

    /// Features encoder
    x_encode: MinMax,

    /// Feature target encoder
    y_encode: MinMax,

    /// Training dataset split
    training_data: (Array2<f64>, Array2<f64>),

    /// Testing data
    testing_data: (Array2<f64>, Array2<f64>),

    /// SGD model with option to add optimizer
    sgd: SGD

}


impl ModelPipeline for HousePrices {
 
    fn register(name: &str) -> Self {

        let temp_x: Array2<f64> = Array2::zeros((0, 0));
        let temp_y: Array2<f64> = Array2::zeros((0, 0));

        HousePrices {
            name: name.to_string(),
            file_path: "data/coin_monero.csv".into(),
            x: temp_x.clone(),
            y: temp_y.clone(),
            x_encode: MinMax::new(),
            y_encode: MinMax::new(),
            training_data: (temp_x.clone(), temp_y.clone()),
            testing_data: (temp_x.clone(), temp_y.clone()),
            sgd: SGD::new(&temp_x, &temp_y, 0.01).unwrap()
        }
    }

}


impl Load for HousePrices {

    fn load(&mut self) {

        println!("Running load step for: {:?}", self.name); 

        let mut file = std::fs::File::open(&self.file_path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();

        let df_select = df.select([
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "population",
            "median_income",
        ]).unwrap();


        let df_target = df.select(["median_house_value"]).unwrap(); 

        self.x = df_select.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

        self.y = df_target.
            to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

    }

}