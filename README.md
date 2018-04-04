# Howto


Install the requirements
```bash
pip install -r requirements.txt
```

Required directory structure
```bash
├── datasets
├── images
├── logs
├── models
├── tools
│   ├── application.py
│   ├── dataset_prepper.py
│   ├── evaluator.py
│   ├── extractor.py
│   ├── loggers.py
│   ├── model_creator.py
│   ├── requirements.txt
└── trained_models
```

Create dataset
```bash
python dataset_prepper.py
```

Create model
```bash
python model_creator.py
```

Start training models
```bash
python application.py
```
