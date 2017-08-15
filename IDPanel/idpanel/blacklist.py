# labels to ignore during training
# this allows for gathering of information for labels which have very sparse labels
# but not including them in classification
labels_to_ignore = [
    "madnesspro"
]


# We don't really want to train on any features that contain these strings
# Based at least partially on user preference, so change as you wish
# Overly aggressive blacklist may result in
feature_blacklist = [
    #".php",
    ".htaccess",
    #".gz",
]
