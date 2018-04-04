# Howto

Install the requirements
```bash
pip install -r requirements.txt
```

Create following file structure:
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

Use this script to create dataset from images in `images/` folder
```bash
python dataset_prepper.py
```

Use this script that creates model in `models/` folder
```bash
python model_creator.py
```

Use this script to start training models on dataset in `datasets/`
```bash
python application.py
```
