import tensorflow as tf

def get_category_subgraph(embeddings, relations, device):
  # Input
  categories_input = list()
  for category in relations.keys():
    category_tensor = tf.constant(relations[category], name="category_input")
    categories_input.append(category_tensor)

  # Graph
  with tf.device(device):
    # Look up embeddings for categories
    categories = list()
    for category in categories_input:
      categories.append(tf.nn.embedding_lookup(embeddings, category, name="category_embedding"))

    # Categorical knowledge additional term
    with tf.name_scope("category_loss") as scope:
      losses = list()
      for category in categories:
        centroid = tf.matmul(tf.ones_like(category), tf.diag(tf.reduce_mean(category, 0)), name="centroid")
        losses.append(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
          category, centroid), 2), 1), name="category_sqrt"), name="mean_loss_in_category"))

      # Building join objective, normalizing second objective by the amount of categories
      category_loss = tf.mul(tf.constant(1.0), tf.add_n(losses), name="category_loss")
      category_loss_summary = tf.scalar_summary("category_loss", category_loss)
  return category_loss, category_loss_summary
