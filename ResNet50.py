from tensorflow import keras
num_classes = 10
def identity_block(input_tensor,
                   kernel_size,
                   num_filters,
                   stage,
                   block):


    filters1, filters2, filters3 = num_filters
    bn_axis = 3
    conv_name_base = 'conv_stage_' + str(stage)+"_block_" + block + '_part_'
    bn_name_base = 'bn_stage_' + str(stage)+"_block_" + block + '_part_'

    x = keras.layers.Conv2D(filters1, (1, 1),name=conv_name_base + 'a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size,padding='same',name=conv_name_base + 'b')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1),name=conv_name_base + 'c')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'c')(x)

    x = keras.layers.add([x, input_tensor], name = 'add_stage_' + str(stage)+"_block_" + block)
    x = keras.layers.Activation('relu')(x)
    return x




# a identity block which has a convolutional layer in the shortcut path
def conv_block(input_tensor,
               kernel_size,
               num_filters,
               stage,
               block,
               strides=(2,2)):

    bn_axis = 3
    filters1, filters2, filters3 = num_filters
    conv_name_base = 'conv_stage_' + str(stage)+"_block_" + block + '_part_'
    bn_name_base = 'bn_stage_' + str(stage)+"_block_" + block + '_part_'

    x = keras.layers.Conv2D(filters1, (1, 1), strides=strides,name=conv_name_base + 'a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
    x = keras.layers.Activation('relu')(x)


    x = keras.layers.Conv2D(filters2, kernel_size, padding='same',name=conv_name_base + 'b')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
    x = keras.layers.Activation('relu')(x)


    x = keras.layers.Conv2D(filters3, (1, 1),name=conv_name_base + 'c')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'c')(x)

    shortcut = keras.layers.Conv2D(filters3, (1,1), strides = strides,name=conv_name_base + 'shortcut')(input_tensor)
    shortcut = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'shortcut')(shortcut)


    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x

input_tensor = keras.Input((224,224,3))

x = keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
x = keras.layers.Conv2D(64, (7, 7),strides=(2, 2),padding='valid',name='stage_1_conv_1')(x)
x = keras.layers.BatchNormalization(axis=3, name='stage1_bn1')(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.ZeroPadding2D(padding=(1, 1), name='stage_1_pool1_pad')(x)
x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)


#stage 2
x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

#stage 3
x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

#stage 4
x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

#stage 5
x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

#final stage

x = keras.layers.GlobalAveragePooling2D()(x)
output = keras.layers.Dense(num_classes, activation ="softmax", name = "fully_conected_layers")(x)

model = keras.Model(input_tensor, output)

print(model.summary())
