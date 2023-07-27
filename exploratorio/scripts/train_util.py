from aim import Run, Image
from dataclasses import dataclass, field

empty_dict_factory = lambda: dict()

@dataclass(frozen=True)
class Artifacts:
    metrics: dict = field(default_factory=empty_dict_factory)
    graphics: dict = field(default_factory=empty_dict_factory)
    other: dict = field(default_factory=empty_dict_factory)

    def track(self, run: Run):
        for metric in self.metrics:
            run.track(self.metrics[metric], name=metric)
        for graphic in self.graphics:
            img = Image(self.graphics[graphic])
            run.track(img, name=graphic)
