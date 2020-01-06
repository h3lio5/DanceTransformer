import os 
import sys
# sys.path.append("../")
print(os.getcwd())
from tfp.config.config import SPLIT_JSON_LOC
# from tfp.utils.Split import Split


def test_config_for_split():
	print(SPLIT_JSON_LOC)

	


if __name__ == "__main__":
	print("Running manual test")
	test_config_for_split()
