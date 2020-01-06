# Transformers For Pose

## Installation

```bash
git clone https://github.com/orionpax00/transformers-for-pose.git
cd transformers-for-pose
virtualenv env
pip install -e . ## for those who want to develop
```

## Projects Structure
```bash
|-- subjects
  |-- 01
    |-- 01_01.asf
    |-- 01_01.amc
    |-- 01_02.amc
    .
    .
|-- npsubjects #this folder will be generated when you run the tfp/resources/amc_to_numpy.py
  |-- 01
    |-- 01_01.npy
    |-- 01_01.npy
    .
    .
|-- tfp
  |-- config
    |--
  |-- resources
    |--
  |-- utils
    |--
  |-- models
    |--
|-- main.py ##train model
```

## RUN
```bash
python main.py salsa --first_time True --seq_len 100 --overlap 0
```
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgement

* [CalciferZh/AMCParser](https://github.com/CalciferZh/AMCParser/blob/master/amc_parser.py)
