# template_manager.py full file
import cv2
from pathlib import Path
import logging

logger = logging.getLogger("TastyTravelsBot")

class TemplateManager:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.categories = {'generators': [], 'orders': []}
        self.energy_template = None
        self.energy_mask = None
        self.go_template = None  
        self.go_mask = None
        self.load_all()

    def _create_mask(self, template):
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return mask

    def load_all(self):
        # Energy
        energy_path = self.base_path / "energy.png"
        if energy_path.exists():
            self.energy_template = cv2.imread(str(energy_path))
            self.energy_mask = self._create_mask(self.energy_template)

        # GO Button from board folder
        go_path = self.base_path / "board" / "go.png"
        if go_path.exists():
            self.go_template = cv2.imread(str(go_path))
            self.go_mask = self._create_mask(self.go_template)
            logger.info("GO template and mask loaded.")

        # Generators & Orders
        for category in ['generators', 'orders']:
            comp_path = self.base_path / category / "components"
            comp_path.mkdir(parents=True, exist_ok=True)
            for img_path in comp_path.glob("*.png"):
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.categories[category].append({
                        'id': img_path.stem,
                        'img': img,
                        'mask': self._create_mask(img)
                    })