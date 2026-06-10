use dendritic::optimizer::model::*; 
use dendritic_ml_models::coca_cola_stock::CocaColaStockModel;
use dendritic_ml_models::breast_cancer::BreastCancerModel;
use dendritic_ml_models::iris::*;
use dendritic_ml_models::titanic::*;

fn main() {

    let mut model = TitanicModel::new();
    model.load_data(0.3);
    model.train();
    
}
