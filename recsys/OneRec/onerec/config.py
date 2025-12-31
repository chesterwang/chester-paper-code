"""
Global configuration for OneRec model
"""
import threading


class OneRecConfig:
    """
    Global configuration class for OneRec model parameters
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OneRecConfig, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'OneRecConfig':
        """Get singleton instance of OneRecConfig"""
        return cls()

    def __init__(self):
        if not self._initialized:

            # base info
            self.gender_dim = 64
            self.age_dim = 64

            # Vocabulary sizes
            self.uid_vocab_size = 1000
            self.vid_vocab_size = 1000
            self.aid_vocab_size = 300

            # Dimension parameters
            self.vid_dim = 32
            self.aid_dim = 32
            self.tag_dim = 32
            self.ts_dim = 32
            self.playtime_dim = 32
            self.dur_dim = 32
            self.label_dim = 32

            # Other model parameters
            self.num_query_tokens = 4
            self.num_qformer_layers = 4
            self.num_rq_layers = 3
            self.codebook_size = 256
            self.multimodal_hidden_dim = 512
            self.qformer_hidden_dim = 512

            self.encoder_model_dim = 512
            self.num_encoder_layers = 6
            self.max_seq_len = 2500
            self.encoder_num_heads = 8
            self.encoder_ff_dim = 2048

            self.decoder_model_dim = 512
            self.num_decoder_layers = 6
            self.decoder_num_heads = 8
            self.decoder_ff_dim = 2048

            self.num_experts = 8
            self.top_k = 2
            self.user_dim = 512
            self.item_dim = 512
            self.num_objectives = 5
            self.num_industrial_objectives = 3

            ##PreferenceScoreTower
            # self.hidden_dim: int = 1024,
            self.num_objectives: int = 5,  # ctr, lvtr, ltr, vtr, etc.
            self.tower_hidden_dim: int = 512 

            self._initialized = True

    def update_config(self, **kwargs):
        """
        Update configuration parameters
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")


# Global configuration instance
config = OneRecConfig()