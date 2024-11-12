from ..step import LocalStep


class ColumnUpdater(LocalStep):
    """
    Simple class to create new columns from existing columns in a dataset.
    Existing columns are not modified.

    Args:
        keys_mapping (dict): Dictionary with "from:to" mapping.
    """

    def __init__(self, keys_mapping: dict, **kwargs):
        super().__init__(**kwargs)
        self.keys_mapping = keys_mapping

    def process_item(self, item, index, datasets, **kwargs):
        for from_key, to_key in self.keys_mapping.items():
            item[to_key] = item[from_key]
        return item


class FlattenList(LocalStep):
    """
    Class to join a list of strings into a single string.
    """

    def __init__(self, input_key, output_key, string_join=", ", **kwargs):
        """
        Args:
            input_key (str): Key to the list of strings.
            output_key (str): Key to store the joined string.
            string_join (str): String to join the list of strings. Defaults to ", ".
        """
        super().__init__(**kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.string_join = string_join

    def process_item(self, item, index, datasets, **kwargs):
        item[self.output_key] = self.string_join.join(item[self.input_key])
        return item


class UpdateField(LocalStep):
    """
    Class to update a field in the dataset with a new value.
    """

    def __init__(self, input_key: str, value, **kwargs):
        """
        Args:
            input_key (str): example key to change.
            value: New value to set for the field.
        """
        super().__init__(**kwargs)
        self.input_key = input_key
        self.value = value

    def process_item(self, item, index, datasets, **kwargs):
        item[self.input_key] = self.value
        return item
