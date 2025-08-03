export interface FaceVerificationResult {
  isRecognized: boolean;
  confidence: number;
  error?: string;
}

export interface FaceVerificationStats {
  successRate: number;
  averageConfidence: number;
  totalSamples: number;
}

export class FaceVerificationService {
  private videoRef: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private isInitialized = false;
  private referenceImageData: string | null = null;

  async initialize(videoElement: HTMLVideoElement, referenceImage?: string): Promise<boolean> {
    try {
      this.videoRef = videoElement;
      this.referenceImageData = referenceImage || null;
      
      // Request camera access
      this.stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 }, 
          height: { ideal: 480 },
          facingMode: 'user'
        } 
      });
      
      this.videoRef.srcObject = this.stream;
      await new Promise((resolve) => {
        this.videoRef!.onloadedmetadata = resolve;
      });
      
      this.isInitialized = true;
      return true;
    } catch (error) {
      console.error('Failed to initialize camera:', error);
      this.isInitialized = false;
      return false;
    }
  }

  async verifyFace(): Promise<FaceVerificationResult> {
    if (!this.isInitialized || !this.videoRef) {
      return { isRecognized: false, confidence: 0, error: 'Service not initialized' };
    }

    try {
      // For demo purposes, we'll simulate face verification
      // In a real implementation, you'd use face-api.js or similar library
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        return { isRecognized: false, confidence: 0, error: 'Failed to create canvas context' };
      }

      canvas.width = this.videoRef.videoWidth;
      canvas.height = this.videoRef.videoHeight;
      ctx.drawImage(this.videoRef, 0, 0);

      // Simulate face detection and recognition
      // In reality, you'd analyze the image for faces and compare with reference
      const simulatedConfidence = 0.75 + Math.random() * 0.2; // 75-95% confidence
      const isRecognized = simulatedConfidence > 0.6;

      return {
        isRecognized,
        confidence: simulatedConfidence,
      };
    } catch (error) {
      console.error('Face verification error:', error);
      return { isRecognized: false, confidence: 0, error: 'Verification failed' };
    }
  }

  async verifyFaceContinuously(durationMs: number = 3000): Promise<FaceVerificationStats> {
    const results: FaceVerificationResult[] = [];
    const startTime = Date.now();
    const interval = 100; // Check every 100ms

    return new Promise((resolve) => {
      const checkInterval = setInterval(async () => {
        const result = await this.verifyFace();
        results.push(result);

        if (Date.now() - startTime >= durationMs) {
          clearInterval(checkInterval);
          
          const successfulVerifications = results.filter(r => r.isRecognized);
          const confidenceScores = successfulVerifications.map(r => r.confidence);
          
          resolve({
            successRate: successfulVerifications.length / results.length,
            averageConfidence: confidenceScores.length > 0 
              ? confidenceScores.reduce((a, b) => a + b) / confidenceScores.length 
              : 0,
            totalSamples: results.length
          });
        }
      }, interval);
    });
  }

  getCurrentFrame(): string | null {
    if (!this.videoRef || !this.isInitialized) return null;
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return null;
    
    canvas.width = this.videoRef.videoWidth;
    canvas.height = this.videoRef.videoHeight;
    ctx.drawImage(this.videoRef, 0, 0);
    
    return canvas.toDataURL('image/jpeg', 0.8);
  }

  cleanup(): void {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    
    if (this.videoRef) {
      this.videoRef.srcObject = null;
    }
    
    this.videoRef = null;
    this.isInitialized = false;
  }

  isReady(): boolean {
    return this.isInitialized && this.videoRef !== null;
  }
}