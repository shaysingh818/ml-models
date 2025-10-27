use dendritic::optimizer::model::*; 
use dendritic_ml_models::coca_cola_stock::CocaColaStockModel;


fn main() {

    let mut model = CocaColaStockModel::register("coca_cola");
    model.load();
    model.transform();
    model.train();
    model.inference();
    

}
