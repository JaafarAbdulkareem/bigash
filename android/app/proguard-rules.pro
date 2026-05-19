# ──────────────────────────────────────────────────────────────────────
# Bigash — ProGuard / R8 rules
#
# The TensorFlow Lite runtime uses reflection to instantiate ops,
# delegates and internal helpers. R8's full-mode obfuscation strips
# these classes and triggers crashes the moment `Interpreter.fromAsset`
# is called in a release build (no warning in debug — only in release).
# ──────────────────────────────────────────────────────────────────────

# Keep all TensorFlow Lite classes
-keep class org.tensorflow.lite.** { *; }
-keep interface org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# Keep the GPU delegate even though we don't currently use it — the
# tflite_flutter plugin pulls it in via reflection.
-keep class org.tensorflow.lite.gpu.** { *; }
-dontwarn org.tensorflow.lite.gpu.**

# Keep the NNAPI delegate (Android Neural Networks API acceleration)
-keep class org.tensorflow.lite.nnapi.** { *; }
-dontwarn org.tensorflow.lite.nnapi.**

# tflite_flutter plugin native bridge
-keep class com.tflite.** { *; }
-dontwarn com.tflite.**

# CameraX + camera plugin reflection
-keep class androidx.camera.** { *; }
-dontwarn androidx.camera.**

# flutter_tts uses Android TextToSpeech callbacks that are invoked
# by name through reflection.
-keep class com.tundralabs.fluttertts.** { *; }

# permission_handler reflection
-keep class com.baseflow.permissionhandler.** { *; }

# Generic safety: keep our serializable data classes (defensive — Equatable
# uses reflection internally on some platforms).
-keepclassmembers class * implements java.io.Serializable {
    static final long serialVersionUID;
    private static final java.io.ObjectStreamField[] serialPersistentFields;
    private void writeObject(java.io.ObjectOutputStream);
    private void readObject(java.io.ObjectInputStream);
    java.lang.Object writeReplace();
    java.lang.Object readResolve();
}
