from tensorflow import keras


num_classes = 10

def conv2D_func(input_tensor,
                num_filters,
                num_rows,
                num_cols,
                padding="same",
                strides=(1, 1)):
    x = keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=(num_rows, num_cols),
        strides=strides,
        padding=padding)(input_tensor)

    x = keras.layers.BatchNormalization(axis=-1,
                                        scale=False)(x)

    x = keras.layers.Activation('relu')(x)

    return x



image_input = keras.Input((299, 299, 3))

# Before inception modules
x = conv2D_func(image_input, 32, 3, 3, strides=(2, 2), padding="valid")
x = conv2D_func(x, 32, 3, 3, padding='valid')
x = conv2D_func(x, 64, 3, 3)
x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)


x = conv2D_func(x, 80, 1, 1, padding='valid')
x = conv2D_func(x, 192, 3, 3, padding='valid')
x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name="Pre_Inception_module")(x)

# inception module A Nº1
branch_a = conv2D_func(x, 64, 1, 1)

branch_b = conv2D_func(x, 48, 1, 1)
branch_b = conv2D_func(branch_b, 64, 5, 5)

branch_c = conv2D_func(x, 64, 1, 1)
branch_c = conv2D_func(branch_c, 96, 3, 3)
branch_c = conv2D_func(branch_c, 96, 3, 3)

branch_d = keras.layers.AveragePooling2D((3, 3),
                                         strides=(1, 1),
                                         padding='same')(x)
branch_d = conv2D_func(branch_d, 32, 1, 1)

x = keras.layers.concatenate(
    [branch_a, branch_b, branch_c, branch_d],
    axis=-1,
    name='Module_A_1')

# Module A Nº2: 35 x 35 x 288
branch_a = conv2D_func(x, 64, 1, 1)

branch_b = conv2D_func(x, 48, 1, 1)
branch_b = conv2D_func(branch_b, 64, 5, 5)

branch_c = conv2D_func(x, 64, 1, 1)
branch_c = conv2D_func(branch_c, 96, 3, 3)
branch_c = conv2D_func(branch_c, 96, 3, 3)

branch_d = keras.layers.AveragePooling2D((3, 3),
                                         strides=(1, 1),
                                         padding='same')(x)
branch_d = conv2D_func(branch_d, 64, 1, 1)
x = keras.layers.concatenate(
    [branch_a, branch_b, branch_c, branch_d],
    axis=-1,
    name='Module_A_2')

# Grid size reduction
branch_a = conv2D_func(x, 384, 3, 3, strides=(2, 2), padding='valid')

branch_b = conv2D_func(x, 64, 1, 1)
branch_b = conv2D_func(branch_b, 96, 3, 3)
branch_b = conv2D_func(branch_b, 96, 3, 3, strides=(2, 2), padding='valid')

branch_c = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
x = keras.layers.concatenate(
    [branch_a, branch_b, branch_c],
    axis=-1,
    name='Grid_Size_redction_1')

# module B Nº1: 17 x 17 x 768
branch_a = conv2D_func(x, 192, 1, 1)

branch_b = conv2D_func(x, 128, 1, 1)
branch_b = conv2D_func(branch_b, 128, 1, 7)
branch_b = conv2D_func(branch_b, 192, 7, 1)

branch_c = conv2D_func(x, 128, 1, 1)
branch_c = conv2D_func(branch_c, 128, 7, 1)
branch_c = conv2D_func(branch_c, 128, 1, 7)
branch_c = conv2D_func(branch_c, 128, 7, 1)
branch_c = conv2D_func(branch_c, 192, 1, 7)

branch_d = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_d = conv2D_func(branch_d, 192, 1, 1)
x = keras.layers.concatenate(
    [branch_a, branch_b, branch_c, branch_d],
    axis=-1,
    name='Module_B_1')

# Module B Nº 2
branch_a = conv2D_func(x, 192, 1, 1)

branch_b = conv2D_func(x, 160, 1, 1)
branch_b = conv2D_func(branch_b, 160, 1, 7)
branch_b = conv2D_func(branch_b, 192, 7, 1)

branch_c = conv2D_func(x, 160, 1, 1)
branch_c = conv2D_func(branch_c, 160, 7, 1)
branch_c = conv2D_func(branch_c, 160, 1, 7)
branch_c = conv2D_func(branch_c, 160, 7, 1)
branch_c = conv2D_func(branch_c, 192, 1, 7)

branch_d = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_d = conv2D_func(branch_d, 192, 1, 1)

x = keras.layers.concatenate(
    [branch_a, branch_b, branch_c, branch_d],
    axis=-1,
    name='Module_B_2')

# Module B Nº 3
branch_a = conv2D_func(x, 192, 1, 1)

branch_b = conv2D_func(x, 160, 1, 1)
branch_b = conv2D_func(branch_b, 160, 1, 7)
branch_b = conv2D_func(branch_b, 192, 7, 1)

branch_c = conv2D_func(x, 160, 1, 1)
branch_c = conv2D_func(branch_c, 160, 7, 1)
branch_c = conv2D_func(branch_c, 160, 1, 7)
branch_c = conv2D_func(branch_c, 160, 7, 1)
branch_c = conv2D_func(branch_c, 192, 1, 7)

branch_d = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_d = conv2D_func(branch_d, 192, 1, 1)

x = keras.layers.concatenate(
    [branch_a, branch_b, branch_c, branch_d],
    axis=-1,
    name='Module_B_3')

# Module B Nº 4
branch_a = conv2D_func(x, 192, 1, 1)

branch_d = conv2D_func(x, 192, 1, 1)
branch_d = conv2D_func(branch_d, 192, 1, 7)
branch_d = conv2D_func(branch_d, 192, 7, 1)

branch_c = conv2D_func(x, 192, 1, 1)
branch_c = conv2D_func(branch_c, 192, 7, 1)
branch_c = conv2D_func(branch_c, 192, 1, 7)
branch_c = conv2D_func(branch_c, 192, 7, 1)
branch_c = conv2D_func(branch_c, 192, 1, 7)

branch_d = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_d = conv2D_func(branch_d, 192, 1, 1)

x = keras.layers.concatenate(
    [branch_a, branch_d, branch_c, branch_d],
    axis=-1,
    name='Module_B_4')

# Grid Size Reduction

branch_a = conv2D_func(x, 192, 1, 1)
branch_a = conv2D_func(branch_a, 320, 3, 3,
                       strides=(2, 2), padding='valid')

branch_b = conv2D_func(x, 192, 1, 1)
branch_b = conv2D_func(branch_b, 192, 1, 7)
branch_b = conv2D_func(branch_b, 192, 7, 1)
branch_b = conv2D_func(branch_b, 192, 3, 3, strides=(2, 2), padding='valid')

branch_c = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

x = keras.layers.concatenate(
    [branch_a, branch_b, branch_c],
    axis=-1,
    name='Grid_Size_Reduction_2')

# Module C Nº 1
branch_a = conv2D_func(x, 320, 1, 1)

branch_b = conv2D_func(x, 384, 1, 1)
branch_b_1 = conv2D_func(branch_b, 384,1,3)
branch_b_2 = conv2D_func(branch_b,384,3,1)
branch_b = x = keras.layers.concatenate([branch_b_1, branch_b_2],axis=-1)

branch_c = conv2D_func(x,448,1,1)
branch_c = conv2D_func(branch_c,384,3,3)
branch_c_1 = conv2D_func(branch_c,384,1,3)
branch_c_2 = conv2D_func(branch_c,384,3,1)
branch_c = x = keras.layers.concatenate([branch_c_1, branch_c_2],axis=-1)

branch_d = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_d = conv2D_func(x,192,1,1)

x = keras.layers.concatenate(
    [branch_a, branch_b, branch_c, branch_d],
    axis=-1,
    name='Module_C_1')

# Module C Nº 2
branch_a = conv2D_func(x, 320, 1, 1)

branch_b = conv2D_func(x, 384, 1, 1)
branch_b_1 = conv2D_func(branch_b, 384,1,3)
branch_b_2 = conv2D_func(branch_b,384,3,1)
branch_b = x = keras.layers.concatenate([branch_b_1, branch_b_2],axis=-1)

branch_c = conv2D_func(x,448,1,1)
branch_c = conv2D_func(branch_c,384,3,3)
branch_c_1 = conv2D_func(branch_c,384,1,3)
branch_c_2 = conv2D_func(branch_c,384,3,1)
branch_c = x = keras.layers.concatenate([branch_c_1, branch_c_2],axis=-1)

branch_d = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_d = conv2D_func(x,192,1,1)

x = keras.layers.concatenate(
    [branch_a, branch_b, branch_c, branch_d],
    axis=-1,
    name='Module_C_2')

#Final part
output = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
output = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(output)



model = keras.Model(image_input, output)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['acc'])

print(model.summary())


