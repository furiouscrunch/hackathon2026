package com.yourapp.capture

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.os.Build
import android.util.Log
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Mic capture for speakerphone call analysis.
 *
 * Emits 1-second mono windows of float samples in [-1, 1] at 16 kHz,
 * 0.5 s hop. Feed each window straight to your spectrogram code.
 *
 * Strategy for beating echo cancellation:
 *   1. Try every audio source x every physical built-in mic.
 *   2. Score each combination by how loud the audio actually is.
 *   3. Keep the loudest. AEC leaves a quiet residual, so louder = less cancelled.
 *   4. Force-disable any AEC/NS/AGC attached to the session.
 *   5. Normalise the output so a faint-but-present signal is still usable.
 *
 * CALIBRATE WHILE THE CALLER IS TALKING. Call start() a few seconds into
 * the call, not before it connects, or every source scores as silence and
 * the pick is meaningless.
 *
 * MANIFEST:
 *   <uses-permission android:name="android.permission.RECORD_AUDIO"/>
 *   <uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>
 *   <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE"/>
 *   <service android:name=".CaptureService"
 *            android:foregroundServiceType="microphone"/>
 *
 * Must be started from a running foreground service with that type, or
 * Android 11+ returns silence with no error.
 */
class CallAudioCapture(
    private val context: Context,
    private val normalise: Boolean = true,
    private val onWindow: (FloatArray) -> Unit
) {

    companion object {
        const val SAMPLE_RATE = 16000
        const val WINDOW_SAMPLES = 16000   // 1.0 s
        const val HOP_SAMPLES = 8000       // 0.5 s
        private const val TAG = "CallAudioCapture"
        private const val PROBE_MS = 250L
        private const val TARGET_RMS = 0.08f   // normalisation target
        private const val MAX_GAIN = 12f
    }

    data class Pick(val source: Int, val deviceId: Int, val rms: Double)

    @Volatile private var running = false
    private var worker: Thread? = null
    var chosen: Pick? = null
        private set

    // ---------------------------------------------------------------- sources

    private fun candidateSources(): List<Int> {
        val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val unprocessedOk = am.getProperty(
            AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED
        ) == "true"

        val list = mutableListOf<Int>()
        if (unprocessedOk && Build.VERSION.SDK_INT >= 24) {
            list.add(MediaRecorder.AudioSource.UNPROCESSED)
        }
        list.add(MediaRecorder.AudioSource.MIC)
        list.add(MediaRecorder.AudioSource.VOICE_RECOGNITION)
        list.add(MediaRecorder.AudioSource.CAMCORDER)
        // VOICE_COMMUNICATION excluded on purpose: it force-applies AEC, which
        // deletes the caller's voice coming out of your own speaker.
        return list
    }

    /** Physical built-in mics. Some phones expose bottom / top / back separately. */
    private fun builtInMics(): List<AudioDeviceInfo?> {
        val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val mics = am.getDevices(AudioManager.GET_DEVICES_INPUTS)
            .filter { it.type == AudioDeviceInfo.TYPE_BUILTIN_MIC }
        // null = "let the system choose", always worth trying as a baseline
        return listOf(null) + mics
    }

    // ---------------------------------------------------------------- opening

    @SuppressLint("MissingPermission")
    private fun open(source: Int, device: AudioDeviceInfo?): AudioRecord? {
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf == AudioRecord.ERROR || minBuf == AudioRecord.ERROR_BAD_VALUE) return null

        val rec = try {
            AudioRecord(
                source, SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                minBuf * 4
            )
        } catch (e: Exception) {
            Log.w(TAG, "construct failed src=$source", e); return null
        }
        if (rec.state != AudioRecord.STATE_INITIALIZED) { rec.release(); return null }

        if (device != null && !rec.setPreferredDevice(device)) {
            // device not selectable for this source; not fatal, just means
            // we're recording from whatever the system picked
            Log.d(TAG, "preferred device ${device.id} rejected for src=$source")
        }
        disableEffects(rec.audioSessionId)
        return rec
    }

    /**
     * OEMs attach AEC/NS/AGC to your session without asking. You can't stop
     * them attaching, but you can grab the handle and switch them off. This is
     * often the difference between recording the caller and recording nothing.
     * Effects applied inside the HAL are below this and cannot be reached.
     */
    private fun disableEffects(sessionId: Int) {
        try { if (AcousticEchoCanceler.isAvailable())
            AcousticEchoCanceler.create(sessionId)?.enabled = false } catch (_: Exception) {}
        try { if (NoiseSuppressor.isAvailable())
            NoiseSuppressor.create(sessionId)?.enabled = false } catch (_: Exception) {}
        try { if (AutomaticGainControl.isAvailable())
            AutomaticGainControl.create(sessionId)?.enabled = false } catch (_: Exception) {}
    }

    // --------------------------------------------------------------- probing

    private fun probe(rec: AudioRecord): Double {
        val buf = ShortArray((SAMPLE_RATE * PROBE_MS / 1000).toInt())
        return try {
            rec.startRecording()
            Thread.sleep(PROBE_MS)
            var read = 0
            while (read < buf.size) {
                val n = rec.read(buf, read, buf.size - read)
                if (n <= 0) break
                read += n
            }
            if (read == 0) return 0.0
            var total = 0.0
            for (i in 0 until read) total += buf[i].toDouble() * buf[i]
            sqrt(total / read)
        } catch (e: Exception) {
            0.0
        } finally {
            try { rec.stop() } catch (_: Exception) {}
        }
    }

    /**
     * Sweeps every source x mic combination and returns the loudest.
     * Takes roughly (sources x mics x 250ms). Run this DURING the call while
     * the caller is speaking.
     */
    fun calibrate(): Pick? {
        var best: Pick? = null
        for (src in candidateSources()) {
            for (dev in builtInMics()) {
                val rec = open(src, dev) ?: continue
                val rms = probe(rec)
                val devId = dev?.id ?: -1
                rec.release()
                Log.i(TAG, "src=$src mic=$devId rms=$rms")
                if (best == null || rms > best!!.rms) best = Pick(src, devId, rms)
            }
        }
        Log.i(TAG, "best -> $best")
        return best
    }

    private fun deviceById(id: Int): AudioDeviceInfo? {
        if (id < 0) return null
        val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        return am.getDevices(AudioManager.GET_DEVICES_INPUTS).firstOrNull { it.id == id }
    }

    // --------------------------------------------------------------- capture

    /** Returns false if every combination came back silent. */
    @SuppressLint("MissingPermission")
    fun start(): Boolean {
        if (running) return true

        val pick = calibrate() ?: return false
        chosen = pick
        if (pick.rms < 20.0) {
            Log.e(TAG, "all combinations near-silent (best rms=${pick.rms}). " +
                    "Either nothing was being said during calibration, or this " +
                    "device blocks mic capture during calls.")
            return false
        }

        val rec = open(pick.source, deviceById(pick.deviceId)) ?: return false
        running = true

        worker = thread(name = "audio-capture") {
            val ring = FloatArray(WINDOW_SAMPLES)
            var filled = 0
            val chunk = ShortArray(2048)

            rec.startRecording()
            while (running) {
                val n = rec.read(chunk, 0, chunk.size)
                if (n <= 0) continue
                var i = 0
                while (i < n) {
                    val take = minOf(WINDOW_SAMPLES - filled, n - i)
                    for (k in 0 until take) ring[filled + k] = chunk[i + k] / 32768.0f
                    filled += take
                    i += take

                    if (filled == WINDOW_SAMPLES) {
                        val out = ring.copyOf()
                        if (normalise) applyGain(out)
                        onWindow(out)
                        System.arraycopy(ring, HOP_SAMPLES, ring, 0, WINDOW_SAMPLES - HOP_SAMPLES)
                        filled = WINDOW_SAMPLES - HOP_SAMPLES
                    }
                }
            }
            try { rec.stop() } catch (_: Exception) {}
            rec.release()
        }
        return true
    }

    /**
     * Scales a quiet window up to a usable level. AEC leaves a low-amplitude
     * residual; this makes it audible and keeps the model's input range stable.
     * It amplifies what survived - it cannot restore what was subtracted away.
     */
    private fun applyGain(w: FloatArray) {
        var sum = 0.0
        for (s in w) sum += s.toDouble() * s
        val rms = sqrt(sum / w.size).toFloat()
        if (rms < 1e-5f) return
        var g = TARGET_RMS / rms
        if (g > MAX_GAIN) g = MAX_GAIN
        if (g < 1f) g = 1f            // never attenuate
        var peak = 0f
        for (s in w) { val a = abs(s); if (a > peak) peak = a }
        if (peak * g > 0.99f) g = 0.99f / peak
        for (i in w.indices) w[i] = w[i] * g
    }

    fun stop() {
        running = false
        worker?.join(1000)
        worker = null
    }
}
