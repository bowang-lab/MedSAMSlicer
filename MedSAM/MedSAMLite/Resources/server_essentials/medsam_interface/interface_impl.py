from .engines import *

engine_classes = {
    'Classic MedSAM': ClassicMedSAM,
    'OpenVino MedSAM': OVMedSAMCore,
    'DAFT MedSAM': DAFTSAMCore,
    # 'Medficient SAM': MedficientSAMCore,
}

class MedSAM_Interface:
    curr_engine = None

    def set_engine(self, engine_name):
        self.curr_engine = engine_classes[engine_name]()
    
    def __getattr__(self, attr):
        return getattr(self.curr_engine, attr)
    
    def __setattr__(self, attr, value):
        if attr == 'curr_engine':
            object.__setattr__(self, attr, value)
        setattr(self.curr_engine, attr, value)
    


