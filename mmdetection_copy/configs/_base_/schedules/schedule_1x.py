# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
    )
#optim_wrapper = dict(
#    type='OptimWrapper',
#    optimizer=dict(
#        type='AdamW',
#        lr=0.001,  # 학습률, 필요에 따라 조정 가능
#        weight_decay=0.01,  # AdamW의 weight decay는 종종 SGD와 다르게 설정됨
#        betas=(0.9, 0.999),  # AdamW 기본 beta 값
#        eps=1e-8
#    )
#)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
