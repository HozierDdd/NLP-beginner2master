import json
from abc import ABC, abstractmethod


class CheckDatasetFormat(ABC):

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        try:
            # Load the dataset
            with open(self.file_path, 'r', encoding='utf-8') as f:
                dataset = [json.loads(line) for line in f]  # save each line as a dictionary, and store all in a list.
            self.dataset = dataset
        except Exception as e:
            print(f"Error occurred while reading dataset: {e}")

    @abstractmethod
    def error_check(self):
        # Implement error check logic here
        pass


#     @abstractmethod
#     def check_header(self, header: list) -> bool:
#         """
#         Check if the header of the dataset is in the correct format.
#
#         :param header: List of column names in the dataset header.
#         :return: True if the header is in the correct format, False otherwise.
#         """
#         pass
#
#     @abstractmethod
#     def check_row(self, row: list) -> bool:
#         """
#         Check if a row in the dataset is in the correct format.
#
#         :param row: List of values in the dataset row.
#         :return: True if the row is in the correct format, False otherwise.
#         """
#         pass
#
#     @abstractmethod
#     def check_dataset(self, dataset: list) -> bool:
#         """
#         Check if the entire dataset is in the correct format.
#
#         :param dataset: List of rows, where each row is a list of values.
#         :return: True if the dataset is in the correct format, False otherwise.
#         """
#         pass
#
#
# # Example subclass implementing the abstract methods
# class MyDatasetChecker(CheckDatasetFormat):
#
#     def check_header(self, header: list) -> bool:
#         # Implement header check logic here
#         expected_header = ["Column1", "Column2", "Column3"]
#         return header == expected_header
#
#     def check_row(self, row: list) -> bool:
#         # Implement row check logic here
#         return all(isinstance(value, (int, float, str)) for value in row)
#
#     def check_dataset(self, dataset: list) -> bool:
#         # Implement dataset check logic here
#         if not self.check_header(dataset[0]):
#             return False
#         for row in dataset[1:]:
#             if not self.check_row(row):
#                 return False
#         return True
#
#
# # Example usage
# checker = MyDatasetChecker()
# dataset = [
#     ["Column1", "Column2", "Column3"],
#     [1, 2.0, "three"],
#     [4, 5.5, "six"]
# ]
#
# print(checker.check_dataset(dataset))  # Output: True or False based on dataset format
