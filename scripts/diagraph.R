library(DiagrammeR)

grViz("
  digraph procedure {
    image[
      label = 'Time-Lapse Images',
      shape = cylinder
    ];
    teacher[
      label = 'Training Dataset',
      shape = hexagon
    ];
    cv[
      label = 'Cross-Validation',
      shape = box
    ];
    classifier[
      label = 'Classifier',
      shape = hexagon
    ];
    selection[
      label = 'Image Selection',
      shape = box
    ];
    alignment[
      label = 'Image-to-Image Alignment',
      shape = box
    ];
    classification[
      label = 'Vegetation Classification',
      shape = box
    ];
    dsm[
      label = 'DSM',
      shape = cylinder
    ];
    sim[
      label = 'Simulated Image',
      shape = hexagon
    ];
    gcp[
      label = 'GCP acquisition',
      shape = box
    ];
    airborne[
      label = 'Airborne/Satellite Image',
      shape = cylinder
    ];
    georec[
      label = 'Georectification',
      shape = box
    ];
    vegemap[
      label = 'Vegetation Classification Map',
      shape = hexagon
    ];
    
    {rank = same; teacher; alignment; dsm; airborne;}
    {rank = same; cv; sim;}
    {rank = same; classifier; classification; gcp;}
    subgraph c {
        teacher -> cv -> classifier
    }
    subgraph g {
        dsm -> sim -> gcp
        airborne -> sim
    }
    subgraph main {
      image -> selection -> alignment -> classification -> georec -> vegemap
    }
    alignment -> teacher
    dsm -> georec
    gcp -> georec
    alignment -> gcp
    classifier -> classification
  }
")
