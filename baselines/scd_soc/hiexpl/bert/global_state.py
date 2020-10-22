class _GlobalStateDict:
    def __init__(self):
        self.states = []
        self.current_layer_id = 0
        self.store_flag = True
        self.activated = False

        self.rel_span_len = 0
        self.total_span_len = 1

    def store_state(self, value):
        if self.activated:
            self.states.append(value)
            self.current_layer_id += 1

    def get_states(self):
        states = self.states[self.current_layer_id]  # [B * H]
        states = states.split(1, 0)
        self.current_layer_id += 1
        return states

    def init_store_states(self):
        self.activated = True
        self.states = []
        self.current_layer_id = 0
        self.store_flag = True

    def init_fetch_states(self):
        self.current_layer_id = 0
        self.store_flag = False


global_state_dict = _GlobalStateDict()
