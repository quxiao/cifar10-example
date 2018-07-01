from predictor import predictor

def main():
    worker = predictor.ImagePredictor()
    worker.load_model(
        custom_model_dir='models/ava-snapshot-model',
        symbol_fn='snapshot-symbol.json',
        weight_fn='snapshot-0030.params',
        label_fn='labels.csv')
    worker.predict('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true')
    worker.predict('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/dog.jpg?raw=true')

if __name__ == '__main__':
    main()