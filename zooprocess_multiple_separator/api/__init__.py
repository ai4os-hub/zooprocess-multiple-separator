from webargs import fields

schema = {
    "separation_coordinates": fields.List(
        fields.List(fields.Int()),
        required=True,
        metadata={'description': """a list containing two other lists of ints: the x and y coordinates of pixels that draw lines on the original image, to separate multiple organisms. This list can be used to subset 2D arrays. For example, to create a black image with white separation lines, one can write:
      import numpy as np
      X = np.zeros(image_shape)
      X[separation_coordinates] = 1
      """}
    ),
    "image_shape": fields.Tuple(
        (fields.Int(),fields.Int()),
        required=True,
        metadata={'description': "the height and width of the original image."}
    ),
    "score": fields.Float(
        required=True,
        metadata={'description': "an estimate of the confidence of the network for the quality of separation; this is very appropximate (and in [0,1])."}
    )
}
