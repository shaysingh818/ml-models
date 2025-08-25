use dendritic::optimizer::model::*; 
use dendritic_ml_models::iris::IrisFlowersModel;


fn main() {
    let model = IrisFlowersModel::register("iris_flowers");
    model.load();
}
