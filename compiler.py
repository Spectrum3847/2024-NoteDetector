populator_dst = metadata.MetadataPopulator.with_model_file('notedetector_edgetpu.tflite')

with open('notedetector.tflite', 'rb') as f:
  populator_dst.load_metadata_and_associated_files(f.read())

populator_dst.populate()
updated_model_buf = populator_dst.get_model_buffer()