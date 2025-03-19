from pathlib import Path
from typing import Optional
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from src.config.config_loader import ConfigLoader

class StorageManager:
    """Manager for handling index storage and retrieval."""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader.get_storage_config()
        self.persist_dir = self.config.get('persist_dir', 'storage')
        
    def save_index(self, index: VectorStoreIndex) -> bool:
        """Save index to disk."""
        try:
            if index is None:
                print("No index provided to save")
                return False
                
            # Create storage directory if it doesn't exist
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            
            # Save index
            index.storage_context.persist(persist_dir=self.persist_dir)
            print(f"Successfully saved index to {self.persist_dir}")
            return True
            
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            return False
            
    def load_index(self) -> Optional[VectorStoreIndex]:
        """Load index from disk."""
        try:
            if not Path(self.persist_dir).exists():
                print(f"No index found at {self.persist_dir}")
                return None
                
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            index = load_index_from_storage(storage_context)
            print("Successfully loaded index from storage")
            return index
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return None 