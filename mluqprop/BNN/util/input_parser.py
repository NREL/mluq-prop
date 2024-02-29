from types import SimpleNamespace

class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def parse_input_deck(file_name: str) -> dict:
    """
    Parses the input deck file and returns a dictionary containing
    the simulation parameters and settings.
    
    Args:
        file_name (str): Path to file to input deck file to be parsed.

    Returns:
        input_dict (dict): Dictionary containing the input parameters from file.
    """
    with open(file_name, 'r') as f:
        input_deck = f.readlines()

    # Remove comments, empty lines, and trailing comments
    input_deck = [line.split('#')[0].strip() for line in input_deck if line.strip() and not line.startswith('#')]

    # Split lines into keys and values, and convert values to int, float or string if possible
    input_dict = {}
    for line in input_deck:
        try:
            key, value = line.split('=')
            value = value.strip()
            # Check if value is enclosed in quotes
            if value.startswith('"') and value.endswith('"'):
                # If the value is a string, remove the quotes and store it
                input_dict[key.strip()] = value[1:-1]
            elif value.isdigit():
                # If the value is an integer, store it as an integer
                input_dict[key.strip()] = int(value)
            else:
                # Try to convert value to a float
                try:
                    input_dict[key.strip()] = float(value)
                except ValueError:
                    # If value is not a number, store it as a string
                    input_dict[key.strip()] = value
        except ValueError:
            print(f"Error: Invalid input line '{line}'")

    # Return the parsed input deck as a namespace.
    return SimpleNamespace(**input_dict)

def check_user_input(simparams, cliargs):
    # Number of layers.
    if cliargs.num_layers != -1:
        simparams.num_layers = cliargs.num_layers
        print("Overriding number of hidden layers with command line argument.")
        print(f"Number of hidden layers: {simparams.num_layers}")
    
    # Hidden dimension.
    if cliargs.hidden_dim != -1:
        simparams.hidden_dim = cliargs.hidden_dim
        print("Overriding hidden layer dimension with command line argument.")
        print(f"Hidden layer dimension: {simparams.hidden_dim}")
    
    # Batch size.
    if cliargs.batch_size != -1:
        simparams.batch_size = cliargs.batch_size
        print("Overriding batch size with command line argument.")
        print(f"Batch size: {simparams.batch_size}")

    # Learning rate.
    if cliargs.learning_rate != -1:
        simparams.learning_rate = cliargs.learning_rate
        print("Overriding learning rate with command line argument.")
        print(f"Learning rate: {simparams.learning_rate}")

    # Checkpoint filepath.
    if cliargs.checkpath != "":
        simparams.checkpath = cliargs.checkpath
        print("Overriding checkpointing filepath with command line argument.")
        print(f"Checkpointing filepath: {simparams.checkpath}")
    
    if cliargs.filename != "":
        simparams.filename = cliargs.filename
        print("Overriding filename with command line argument.")
        print(f"Filename: {simparams.filename}")

    return simparams
