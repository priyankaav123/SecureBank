export interface SpeechRecognitionResult {
  transcript: string;
  confidence: number;
  error?: string;
}

export interface RecordingState {
  isRecording: boolean;
  duration: number;
  error?: string;
}

export class SpeechService {
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private stream: MediaStream | null = null;
  private voskServerUrl = 'ws://localhost:2700';
  private recordingCallback?: (state: RecordingState) => void;
  private recordingStartTime = 0;
  private recordingTimer?: number;

  async initialize(): Promise<boolean> {
    try {
      // Request microphone access
      this.stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      });
      return true;
    } catch (error) {
      console.error('Failed to initialize microphone:', error);
      return false;
    }
  }

  async startRecording(
    durationSeconds: number = 5,
    onStateChange?: (state: RecordingState) => void
  ): Promise<void> {
    if (!this.stream) {
      throw new Error('Speech service not initialized');
    }

    console.log('Starting MediaRecorder setup...');
    this.recordingCallback = onStateChange;
    this.audioChunks = [];
    
    // Try different MIME types for better compatibility
    let mimeType = 'audio/webm;codecs=opus';
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = 'audio/webm';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/mp4';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = ''; // Let browser choose
        }
      }
    }
    
    console.log('Using MIME type:', mimeType);
    
    const options = mimeType ? { mimeType } : {};
    this.mediaRecorder = new MediaRecorder(this.stream, options);

    this.mediaRecorder.ondataavailable = (event) => {
      console.log('Data available, size:', event.data.size);
      if (event.data.size > 0) {
        this.audioChunks.push(event.data);
      }
    };

    this.mediaRecorder.onstart = () => {
      console.log('MediaRecorder started');
    };

    this.mediaRecorder.onstop = () => {
      console.log('MediaRecorder stopped, chunks:', this.audioChunks.length);
    };

    this.mediaRecorder.onerror = (event) => {
      console.error('MediaRecorder error:', event);
    };

    this.recordingStartTime = Date.now();
    
    // Start recording with small intervals to ensure data is captured
    this.mediaRecorder.start(100); // Record in 100ms chunks
    console.log('MediaRecorder.start() called');

    this.recordingTimer = window.setInterval(() => {
      const elapsed = (Date.now() - this.recordingStartTime) / 1000;
      this.recordingCallback?.({
        isRecording: this.mediaRecorder?.state === 'recording',
        duration: elapsed
      });
    }, 100);

    this.recordingCallback?.({
      isRecording: true,
      duration: 0
    });

    // Auto-stop after specified duration
    setTimeout(() => {
      console.log('Auto-stopping recording after', durationSeconds, 'seconds');
      this.stopRecording();
    }, durationSeconds * 1000);
  }

  async stopRecording(): Promise<Blob | null> {
    return new Promise((resolve) => {
      console.log('stopRecording called, mediaRecorder state:', this.mediaRecorder?.state);
      
      if (!this.mediaRecorder) {
        console.log('No mediaRecorder available');
        resolve(null);
        return;
      }

      if (this.mediaRecorder.state === 'inactive') {
        console.log('MediaRecorder already inactive, creating blob from existing chunks');
        if (this.audioChunks.length > 0) {
          const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
          console.log('Created blob from existing chunks, size:', audioBlob.size);
          resolve(audioBlob);
        } else {
          console.log('No audio chunks available');
          resolve(null);
        }
        return;
      }

      if (this.recordingTimer) {
        clearInterval(this.recordingTimer);
        this.recordingTimer = undefined;
      }

      // Set up the stop handler before stopping
      this.mediaRecorder.onstop = () => {
        console.log('MediaRecorder onstop triggered, audioChunks:', this.audioChunks.length);
        
        if (this.audioChunks.length > 0) {
          // Try to determine the correct MIME type
          let mimeType = 'audio/webm';
          if (this.audioChunks[0] && this.audioChunks[0].type) {
            mimeType = this.audioChunks[0].type;
          }
          
          const audioBlob = new Blob(this.audioChunks, { type: mimeType });
          console.log('Created audio blob, size:', audioBlob.size, 'type:', mimeType);
          
          this.recordingCallback?.({
            isRecording: false,
            duration: (Date.now() - this.recordingStartTime) / 1000
          });
          
          resolve(audioBlob);
        } else {
          console.log('No audio chunks to create blob from');
          resolve(null);
        }
      };

      // Add a timeout in case onstop doesn't fire
      const timeout = setTimeout(() => {
        console.log('Stop timeout triggered, forcing blob creation');
        if (this.audioChunks.length > 0) {
          const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
          resolve(audioBlob);
        } else {
          resolve(null);
        }
      }, 1000);

      this.mediaRecorder.addEventListener('stop', () => {
        clearTimeout(timeout);
      }, { once: true });

      console.log('Calling mediaRecorder.stop()');
      this.mediaRecorder.stop();
    });
  }

  async transcribeAudioBlob(audioBlob: Blob): Promise<SpeechRecognitionResult> {
    try {
      console.log('Starting transcription, audioBlob size:', audioBlob.size);

      // Try Web Speech API first (more reliable for browser-based transcription)
      console.log('Attempting Web Speech API transcription...');
      const webSpeechResult = await this.transcribeWithWebSpeech(audioBlob);
      if (webSpeechResult.transcript && webSpeechResult.transcript.trim()) {
        console.log('Web Speech API successful:', webSpeechResult);
        return webSpeechResult;
      }

      console.log('Web Speech API failed or empty, trying Vosk...');
      // Convert WebM to WAV for Vosk compatibility
      const audioBuffer = await this.convertToWav(audioBlob);
      if (!audioBuffer) {
        console.log('Audio conversion failed');
        return { transcript: '', confidence: 0, error: 'Failed to convert audio' };
      }

      // Try Vosk server as fallback
      const voskResult = await this.transcribeWithVosk(audioBuffer);
      if (voskResult.transcript && voskResult.transcript.trim()) {
        console.log('Vosk successful:', voskResult);
        return voskResult;
      }

      console.log('Both transcription methods failed');
      return { 
        transcript: '', 
        confidence: 0, 
        error: 'No transcription available from either method' 
      };
    } catch (error) {
      console.error('Transcription error:', error);
      return { transcript: '', confidence: 0, error: `Transcription failed: ${error}` };
    }
  }

  private async transcribeWithVosk(audioBuffer: ArrayBuffer): Promise<SpeechRecognitionResult> {
    try {
      const ws = new WebSocket(this.voskServerUrl);
      
      return new Promise((resolve) => {
        let partialResult = '';
        const timeout = setTimeout(() => {
          ws.close();
          resolve({ transcript: partialResult, confidence: 0.5, error: 'Vosk timeout' });
        }, 10000);

        ws.onopen = () => {
          // Send configuration
          ws.send(JSON.stringify({ config: { sample_rate: 16000 } }));
          
          // Send audio data in chunks
          const chunkSize = 3200; // 0.2 seconds of audio at 16kHz
          let offset = 0;
          
          const sendChunk = () => {
            if (offset < audioBuffer.byteLength) {
              const chunk = audioBuffer.slice(offset, offset + chunkSize);
              ws.send(chunk);
              offset += chunkSize;
              setTimeout(sendChunk, 100);
            } else {
              // Send EOF
              ws.send(JSON.stringify({ eof: 1 }));
            }
          };
          
          sendChunk();
        };

        ws.onmessage = (event) => {
          try {
            const result = JSON.parse(event.data);
            if (result.partial) {
              partialResult = result.partial;
            }
            if (result.text) {
              clearTimeout(timeout);
              ws.close();
              resolve({ 
                transcript: result.text, 
                confidence: 0.8 
              });
            }
          } catch (e) {
            console.error('Vosk response parsing error:', e);
          }
        };

        ws.onerror = () => {
          clearTimeout(timeout);
          resolve({ transcript: '', confidence: 0, error: 'Vosk connection failed' });
        };
      });
    } catch (error) {
      console.error('Vosk transcription error:', error);
      return { transcript: '', confidence: 0, error: 'Vosk unavailable' };
    }
  }

  private async transcribeWithWebSpeech(audioBlob: Blob): Promise<SpeechRecognitionResult> {
    return new Promise((resolve) => {
      if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        console.log('Web Speech API not supported');
        resolve({ transcript: '', confidence: 0, error: 'Speech recognition not supported' });
        return;
      }

      console.log('Using Web Speech API fallback');
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognition.maxAlternatives = 1;

      let finalTranscript = '';
      let hasResult = false;

      recognition.onresult = (event: any) => {
        console.log('Web Speech API result event:', event);
        let interimTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
            hasResult = true;
          } else {
            interimTranscript += transcript;
          }
        }
        
        console.log('Final transcript:', finalTranscript);
        console.log('Interim transcript:', interimTranscript);

        if (finalTranscript) {
          resolve({
            transcript: finalTranscript,
            confidence: event.results[0][0].confidence || 0.8
          });
        }
      };

      recognition.onerror = (event: any) => {
        console.error('Web Speech API error:', event.error);
        resolve({ transcript: '', confidence: 0, error: `Speech recognition error: ${event.error}` });
      };

      recognition.onend = () => {
        console.log('Web Speech API ended, hasResult:', hasResult);
        if (!hasResult) {
          resolve({ transcript: '', confidence: 0, error: 'No speech detected' });
        }
      };

      // Start recognition directly
      try {
        recognition.start();
        console.log('Web Speech API recognition started');
      } catch (error) {
        console.error('Failed to start Web Speech API:', error);
        resolve({ transcript: '', confidence: 0, error: 'Failed to start speech recognition' });
      }
    });
  }

  private async convertToWav(audioBlob: Blob): Promise<ArrayBuffer | null> {
    try {
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const arrayBuffer = await audioBlob.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Convert to 16-bit PCM WAV format
      const length = audioBuffer.length;
      const sampleRate = 16000;
      const numberOfChannels = 1;
      
      const buffer = new ArrayBuffer(44 + length * 2);
      const view = new DataView(buffer);
      
      // WAV header
      const writeString = (offset: number, string: string) => {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i));
        }
      };
      
      writeString(0, 'RIFF');
      view.setUint32(4, 36 + length * 2, true);
      writeString(8, 'WAVE');
      writeString(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, numberOfChannels, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * numberOfChannels * 2, true);
      view.setUint16(32, numberOfChannels * 2, true);
      view.setUint16(34, 16, true);
      writeString(36, 'data');
      view.setUint32(40, length * 2, true);
      
      // Convert audio data
      const channelData = audioBuffer.getChannelData(0);
      let offset = 44;
      for (let i = 0; i < length; i++) {
        const sample = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
      }
      
      return buffer;
    } catch (error) {
      console.error('Audio conversion error:', error);
      return null;
    }
  }

  cleanup(): void {
    console.log('Cleaning up speech service...');
    
    if (this.recordingTimer) {
      clearInterval(this.recordingTimer);
      this.recordingTimer = undefined;
    }

    if (this.mediaRecorder) {
      console.log('Cleaning up mediaRecorder, state:', this.mediaRecorder.state);
      if (this.mediaRecorder.state === 'recording') {
        console.log('Stopping active recording...');
        this.mediaRecorder.stop();
      }
      // Remove all event listeners
      this.mediaRecorder.ondataavailable = null;
      this.mediaRecorder.onstop = null;
      this.mediaRecorder.onstart = null;
      this.mediaRecorder.onerror = null;
      this.mediaRecorder = null;
    }

    this.audioChunks = [];
    this.recordingCallback = undefined;
    console.log('Speech service cleanup complete');
  }

  // Complete cleanup including stream
  fullCleanup(): void {
    this.cleanup();
    
    if (this.stream) {
      console.log('Stopping media stream...');
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
  }

  async recordAndTranscribe(durationSeconds: number = 5): Promise<SpeechRecognitionResult> {
    await this.startRecording(durationSeconds);
    
    return new Promise((resolve) => {
      const checkRecording = () => {
        if (this.mediaRecorder?.state === 'inactive') {
          this.stopRecording().then(async (audioBlob) => {
            if (audioBlob) {
              const result = await this.transcribeAudioBlob(audioBlob);
              resolve(result);
            } else {
              resolve({ transcript: '', confidence: 0, error: 'No audio recorded' });
            }
          });
        } else {
          setTimeout(checkRecording, 100);
        }
      };
      
      setTimeout(checkRecording, durationSeconds * 1000 + 100);
    });
  }
}