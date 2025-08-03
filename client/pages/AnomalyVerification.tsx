import { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { AlertTriangle, ShieldAlert, Lock, UserCheck, Clock, Mic, MicOff, Video, VideoOff, CheckCircle, XCircle } from "lucide-react";
import { FaceVerificationService, FaceVerificationStats } from '../services/faceVerificationService';
import { SpeechService, SpeechRecognitionResult, RecordingState } from '../services/speechService';

interface AnomalyData {
  anomaly_confidence: number;
  anomaly_reasons: string[];
  recent_events: Array<{
    action: string;
    details: string;
  }>;
}

interface SecurityQuestion {
  question: string;
  expected_context?: string;
}

interface QuestionState {
  isRecording: boolean;
  transcript: string;
  confidence: number;
  error?: string;
  recordingDuration: number;
}

interface FaceVerificationState {
  isActive: boolean;
  isRecognized: boolean;
  confidence: number;
  stats?: FaceVerificationStats;
  error?: string;
}

export default function AnomalyVerification() {
  const navigate = useNavigate();
  const location = useLocation();
  const [anomalyData, setAnomalyData] = useState<AnomalyData | null>(null);
  const [aiQuestions, setAiQuestions] = useState<SecurityQuestion[]>([]);
  const [answers, setAnswers] = useState<string[]>([]);
  const [questionStates, setQuestionStates] = useState<QuestionState[]>([]);
  const [faceVerification, setFaceVerification] = useState<FaceVerificationState>({
    isActive: false,
    isRecognized: false,
    confidence: 0
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingQuestions, setIsLoadingQuestions] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRemaining, setTimeRemaining] = useState(300); // 5 minutes
  
  // Service refs
  const faceServiceRef = useRef<FaceVerificationService | null>(null);
  const speechServiceRef = useRef<SpeechService | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [servicesInitialized, setServicesInitialized] = useState(false);
  const [isRecordingAny, setIsRecordingAny] = useState(false); // Global recording lock

  useEffect(() => {
    // Get anomaly data from location state or session storage
    const anomalyDataFromState = (location.state as any)?.anomalyData;
    const anomalyDataStr = anomalyDataFromState ? JSON.stringify(anomalyDataFromState) : sessionStorage.getItem('anomalyData');
    if (anomalyDataStr) {
      try {
        const data = JSON.parse(anomalyDataStr);
        setAnomalyData(data);
        
        // Generate AI questions based on anomaly data
        generateAIQuestions(data);
      } catch (e) {
        console.error('Failed to parse anomaly data:', e);
        setError('Failed to load security verification data');
        setIsLoadingQuestions(false);
      }
    } else {
      setError('No security verification data found');
      setIsLoadingQuestions(false);
    }

    // Start countdown timer
    const timer = setInterval(() => {
      setTimeRemaining(prev => {
        if (prev <= 1) {
          // Time's up - force logout
          handleTimeout();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [location]);

  const handleTimeout = () => {
    // Force logout and redirect to login
    localStorage.removeItem('authToken');
    sessionStorage.clear(); // This will also clear tempAuthToken
    navigate('/login?message=Session expired due to security verification timeout');
  };

  const generateAIQuestions = async (data: AnomalyData) => {
    try {
      // Try to get the temporary auth token first, then fall back to regular token
      const token = sessionStorage.getItem('tempAuthToken') || localStorage.getItem('authToken');
      console.log('Token exists:', !!token);
      console.log('Token length:', token?.length);
      
      if (!token) {
        setError('Authentication token not found. Please log in again.');
        setIsLoadingQuestions(false);
        return;
      }

      const response = await fetch('http://localhost:5000/api/generate-mixed-security-questions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          anomaly_data: data
        })
      });

      console.log('Response status:', response.status);
      const result = await response.json();
      console.log('Response result:', result);

      if (response.ok && result.success) {
        const questions = result.questions || [];
        setAiQuestions(questions);
        setAnswers(new Array(questions.length).fill(''));
        setQuestionStates(questions.map(() => ({
          isRecording: false,
          transcript: '',
          confidence: 0,
          recordingDuration: 0
        })));
      } else {
        setError(result.error || result.msg || 'Failed to generate security questions');
      }
    } catch (error) {
      console.error('Error generating AI questions:', error);
      setError('Failed to generate security questions');
    } finally {
      setIsLoadingQuestions(false);
    }
  };

  // Initialize services
  useEffect(() => {
    const initializeServices = async () => {
      try {
        // Initialize face verification service
        faceServiceRef.current = new FaceVerificationService();
        speechServiceRef.current = new SpeechService();
        
        if (videoRef.current) {
          const faceInitialized = await faceServiceRef.current.initialize(videoRef.current);
          const speechInitialized = await speechServiceRef.current.initialize();
          
          if (faceInitialized && speechInitialized) {
            setServicesInitialized(true);
            startFaceVerification();
          } else {
            setError('Failed to initialize camera or microphone. Please ensure permissions are granted.');
          }
        }
      } catch (error) {
        console.error('Service initialization error:', error);
        setError('Failed to initialize verification services.');
      }
    };

    initializeServices();

    return () => {
      // Cleanup services
      faceServiceRef.current?.cleanup();
      speechServiceRef.current?.fullCleanup();
      if ((window as any).faceVerificationInterval) {
        clearInterval((window as any).faceVerificationInterval);
      }
    };
  }, []);

  const startFaceVerification = () => {
    if (!faceServiceRef.current?.isReady()) return;

    setFaceVerification(prev => ({ ...prev, isActive: true }));
    
    const verifyInterval = setInterval(async () => {
      if (!faceServiceRef.current?.isReady()) {
        clearInterval(verifyInterval);
        return;
      }

      try {
        const result = await faceServiceRef.current.verifyFace();
        setFaceVerification(prev => ({
          ...prev,
          isRecognized: result.isRecognized,
          confidence: result.confidence,
          error: result.error
        }));
      } catch (error) {
        console.error('Face verification error:', error);
      }
    }, 500); // Check every 500ms

    // Store interval ID for cleanup
    (window as any).faceVerificationInterval = verifyInterval;
  };

  const handleVoiceInput = async (questionIndex: number) => {
    if (!speechServiceRef.current || !faceServiceRef.current?.isReady()) {
      setError('Services not initialized');
      return;
    }

    // Prevent multiple simultaneous recordings
    if (isRecordingAny) {
      console.log('Already recording, ignoring new request');
      return;
    }

    try {
      console.log(`Starting voice input for question ${questionIndex}`);
      setIsRecordingAny(true); // Set global recording lock
      
      // Clean up any previous recording first
      await speechServiceRef.current.cleanup();
      
      // Update question state to recording
      setQuestionStates(prev => {
        console.log(`Starting recording for question ${questionIndex}, preserving other states`);
        return prev.map((state, idx) => {
          if (idx === questionIndex) {
            return {
              ...state,
              isRecording: true,
              error: undefined,
              transcript: '', // Clear previous transcript for this question only
              recordingDuration: 0
            };
          }
          return state; // Keep other states unchanged
        });
      });

      // Start recording with state updates (increased to 8 seconds)
      console.log('Starting recording...');
      await speechServiceRef.current.startRecording(8, (recordingState: RecordingState) => {
        setQuestionStates(prev => {
          return prev.map((state, idx) => {
            if (idx === questionIndex) {
              return {
                ...state,
                isRecording: recordingState.isRecording,
                recordingDuration: recordingState.duration,
                error: recordingState.error
              };
            }
            return state; // Keep other states unchanged
          });
        });
      });

      console.log('Recording started, waiting for completion...');

      // Wait for the recording to complete naturally
      console.log('Waiting for recording to complete...');
      
      // Don't manually stop - let the auto-stop timer handle it
      await new Promise<void>((resolve) => {
        const checkComplete = setInterval(async () => {
          if (!speechServiceRef.current) {
            clearInterval(checkComplete);
            resolve();
            return;
          }

          // Check if MediaRecorder has stopped
          const mediaRecorder = (speechServiceRef.current as any).mediaRecorder;
          if (mediaRecorder && mediaRecorder.state === 'inactive') {
            clearInterval(checkComplete);
            console.log('MediaRecorder is inactive, processing...');
            
            try {
              // Get the recorded audio
              const audioBlob = await speechServiceRef.current.stopRecording();
              console.log('Recording completed, audioBlob size:', audioBlob?.size || 0);

              if (audioBlob && audioBlob.size > 0) {
                console.log('Transcribing audio...');
                // Transcribe the audio
                const transcriptionResult = await speechServiceRef.current.transcribeAudioBlob(audioBlob);
                console.log('Transcription result:', transcriptionResult);

                // Verify face during recording (simplified check)
                const faceStats = await faceServiceRef.current!.verifyFaceContinuously(1000);
                console.log('Face verification stats:', faceStats);

                // Update question state with results - be very specific about preserving other states
                setQuestionStates(prev => {
                  console.log('Previous question states before update:', prev);
                  const newStates = prev.map((state, idx) => {
                    if (idx === questionIndex) {
                      const updatedState = {
                        ...state,
                        isRecording: false,
                        transcript: transcriptionResult.transcript || '',
                        confidence: transcriptionResult.confidence || 0,
                        error: transcriptionResult.error
                      };
                      console.log(`Updating question ${questionIndex} state:`, updatedState);
                      return updatedState;
                    }
                    return state; // Keep other states unchanged
                  });
                  console.log('New question states after update:', newStates);
                  return newStates;
                });

                // Update answers array immediately
                if (transcriptionResult.transcript && transcriptionResult.transcript.trim()) {
                  console.log(`Updating answer for question ${questionIndex}:`, transcriptionResult.transcript);
                  setAnswers(prev => {
                    console.log('Previous answers:', prev);
                    const newAnswers = [...prev];
                    newAnswers[questionIndex] = transcriptionResult.transcript.trim();
                    console.log('Updated answers array:', newAnswers);
                    return newAnswers;
                  });
                }

                // Update face verification stats
                setFaceVerification(prev => ({
                  ...prev,
                  stats: faceStats
                }));
              } else {
                console.log('No valid audio blob received');
                setQuestionStates(prev => {
                  return prev.map((state, idx) => {
                    if (idx === questionIndex) {
                      return {
                        ...state,
                        isRecording: false,
                        error: 'No audio recorded - please try again'
                      };
                    }
                    return state; // Keep other states unchanged
                  });
                });
              }
            } catch (transcriptionError) {
              console.error('Transcription error:', transcriptionError);
              setQuestionStates(prev => {
                return prev.map((state, idx) => {
                  if (idx === questionIndex) {
                    return {
                      ...state,
                      isRecording: false,
                      error: 'Failed to transcribe audio'
                    };
                  }
                  return state; // Keep other states unchanged
                });
              });
            }
            
            resolve();
          }
        }, 100); // Check every 100ms

        // Fallback timeout (increased to 10 seconds)
        setTimeout(() => {
          clearInterval(checkComplete);
          console.log('Timeout reached, processing anyway...');
          resolve();
        }, 10000); // 10 seconds total timeout
      });

    } catch (error) {
      console.error('Voice input error:', error);
      setQuestionStates(prev => {
        return prev.map((state, idx) => {
          if (idx === questionIndex) {
            return {
              ...state,
              isRecording: false,
              error: `Failed to record voice input: ${error}`
            };
          }
          return state; // Keep other states unchanged
        });
      });
    } finally {
      // Always release the global recording lock
      console.log('Releasing recording lock');
      setIsRecordingAny(false);
    }
  };

  const handleAnswerChange = (index: number, answer: string) => {
    setAnswers(prev => {
      const newAnswers = [...prev];
      newAnswers[index] = answer;
      return newAnswers;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    console.log('Submitting form with answers:', answers);
    console.log('Question states:', questionStates);

    // Check if all questions have transcripts (more reliable than answers array)
    const unansweredQuestions = questionStates.filter((state, index) => !state.transcript?.trim());
    if (unansweredQuestions.length > 0) {
      setError(`Please record voice answers for all ${aiQuestions.length} security questions. ${unansweredQuestions.length} questions still need answers.`);
      setIsSubmitting(false);
      return;
    }

    // Ensure answers array is properly populated from transcripts
    const finalAnswers = questionStates.map(state => state.transcript?.trim() || '');
    setAnswers(finalAnswers);

    try {
      const token = sessionStorage.getItem('tempAuthToken') || localStorage.getItem('authToken');
      const response = await fetch('http://localhost:5000/api/verify-mixed-security-answers', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          questions: aiQuestions,
          answers: finalAnswers, // Use the final answers from transcripts
          anomaly_data: anomalyData
        })
      });

      const result = await response.json();

      if (response.ok && result.verified) {
        // Verification successful - restore auth token and clean up
        const tempToken = sessionStorage.getItem('tempAuthToken');
        if (tempToken) {
          localStorage.setItem('authToken', tempToken);
        }
        
        sessionStorage.removeItem('anomalyData');
        sessionStorage.removeItem('tempAuthToken');
        
        navigate('/', { 
          state: { 
            message: result.message || 'Identity verified successfully! Welcome back.',
            type: 'success'
          }
        });
      } else {
        // Verification failed - show impostor message and force logout
        setError(result.message || 'Sorry, an impostor has been detected. Access denied.');
        setTimeout(() => {
          localStorage.removeItem('authToken');
          sessionStorage.clear();
          navigate('/login', {
            state: { 
              message: 'Security verification failed - impostor detected',
              type: 'error'
            }
          });
        }, 4000);
      }
    } catch (err) {
      setError('An error occurred during verification. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-red-600';
    if (confidence >= 60) return 'text-orange-600';
    return 'text-yellow-600';
  };

  return (
    <div className="min-h-screen bg-red-50 flex items-center justify-center p-4">
      <div className="max-w-4xl w-full space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex justify-center">
            <ShieldAlert className="h-16 w-16 text-red-600" />
          </div>
          <h1 className="text-3xl font-bold text-red-800">Security Anomaly Detected</h1>
          <p className="text-red-700">
            Unusual activity has been detected on your account. Please verify your identity to continue.
          </p>
          
          {/* Countdown Timer */}
          <div className="flex items-center justify-center space-x-2 bg-red-100 p-3 rounded-lg border border-red-300">
            <Clock className="h-5 w-5 text-red-600" />
            <span className="font-mono text-lg font-semibold text-red-800">
              Time remaining: {formatTime(timeRemaining)}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Anomaly Details */}
          {anomalyData && (
            <Card className="border-red-200 bg-white">
              <CardHeader>
                <CardTitle className="flex items-center text-red-800">
                  <AlertTriangle className="mr-2 h-6 w-6" />
                  Detected Anomalies
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="font-semibold">
                    Anomaly Confidence: 
                    <span className={`ml-2 ${getConfidenceColor(anomalyData.anomaly_confidence)}`}>
                      {anomalyData.anomaly_confidence.toFixed(2)}%
                    </span>
                  </p>
                </div>

                {anomalyData.anomaly_reasons && anomalyData.anomaly_reasons.length > 0 && (
                  <div>
                    <p className="font-semibold text-red-700 mb-2">Reasons:</p>
                    <ul className="list-disc list-inside space-y-1 text-sm">
                      {anomalyData.anomaly_reasons.map((reason, index) => (
                        <li key={index} className="text-red-800">{reason}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {anomalyData.recent_events && anomalyData.recent_events.length > 0 && (
                  <div>
                    <p className="font-semibold text-gray-700 mb-2">Recent Activity:</p>
                    <div className="bg-gray-50 p-3 rounded max-h-32 overflow-y-auto">
                      <ul className="space-y-1 text-sm">
                        {anomalyData.recent_events.slice(0, 5).map((event, index) => (
                          <li key={index} className="flex justify-between">
                            <span className="font-medium capitalize">
                              {event.action.replace(/_/g, ' ')}
                            </span>
                            <span className="text-gray-600 text-xs">
                              {event.details}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Face Verification */}
          <Card className="border-green-200 bg-white">
            <CardHeader>
              <CardTitle className="flex items-center text-green-800">
                <Video className="mr-2 h-6 w-6" />
                Face Verification
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="relative bg-black rounded-lg overflow-hidden" style={{ aspectRatio: '4/3' }}>
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  className="w-full h-full object-cover"
                  style={{ transform: 'scaleX(-1)' }}
                />
                
                {/* Face verification overlay */}
                <div className="absolute top-4 left-4 flex items-center space-x-2">
                  {faceVerification.isActive ? (
                    faceVerification.isRecognized ? (
                      <>
                        <CheckCircle className="h-6 w-6 text-green-400" />
                        <span className="text-green-400 font-semibold">
                          Face Verified ({(faceVerification.confidence * 100).toFixed(0)}%)
                        </span>
                      </>
                    ) : (
                      <>
                        <XCircle className="h-6 w-6 text-red-400" />
                        <span className="text-red-400 font-semibold">
                          Face Not Recognized
                        </span>
                      </>
                    )
                  ) : (
                    <>
                      <VideoOff className="h-6 w-6 text-gray-400" />
                      <span className="text-gray-400">Initializing...</span>
                    </>
                  )}
                </div>
              </div>
              
              {faceVerification.stats && (
                <div className="text-sm text-gray-600">
                  <p>Success Rate: {(faceVerification.stats.successRate * 100).toFixed(1)}%</p>
                  <p>Average Confidence: {(faceVerification.stats.averageConfidence * 100).toFixed(1)}%</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Security Questions */}
          <Card className="border-blue-200 bg-white">
            <CardHeader>
              <CardTitle className="flex items-center text-blue-800">
                <UserCheck className="mr-2 h-6 w-6" />
                Voice Identity Verification
              </CardTitle>
            </CardHeader>
            <CardContent>
              {isLoadingQuestions ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mr-3"></div>
                  <span>Generating mixed security questions...</span>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-6">
                  <p className="text-sm text-gray-600 mb-4">
                    Please answer the following questions using voice input. This includes 2 personalized security questions and 1 AI-generated question based on your recent activity. Your face will be verified during recording:
                  </p>

                  {aiQuestions.map((question, index) => {
                    const questionState = questionStates[index] || {
                      isRecording: false,
                      transcript: '',
                      confidence: 0,
                      recordingDuration: 0
                    };
                    
                    return (
                      <div key={`question-${index}-${question.question.slice(0,20)}`} className="space-y-3 p-4 border rounded-lg">
                        <div className="flex items-start justify-between">
                          <Label className="font-medium text-base flex-1">
                            {index + 1}. {question.question}
                          </Label>
                          <span className={`text-xs px-2 py-1 rounded-full ${
                            question.type === 'security' 
                              ? 'bg-green-100 text-green-700' 
                              : 'bg-blue-100 text-blue-700'
                          }`}>
                            {question.type === 'security' ? 'Security Question' : 'AI Question'}
                          </span>
                        </div>
                        
                        <div className="flex items-center space-x-3">
                          <Button
                            type="button"
                            onClick={() => handleVoiceInput(index)}
                            disabled={questionState.isRecording || isSubmitting || !servicesInitialized || isRecordingAny}
                            className={`flex items-center space-x-2 ${
                              questionState.isRecording ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'
                            }`}
                          >
                            {questionState.isRecording ? (
                              <>
                                <MicOff className="h-4 w-4" />
                                <span>Recording... {questionState.recordingDuration.toFixed(1)}s / 8s</span>
                              </>
                            ) : isRecordingAny ? (
                              <>
                                <Mic className="h-4 w-4" />
                                <span>Please wait...</span>
                              </>
                            ) : (
                              <>
                                <Mic className="h-4 w-4" />
                                <span>Record Answer (8s)</span>
                              </>
                            )}
                          </Button>
                          
                          {questionState.transcript && (
                            <div className="flex items-center space-x-2 text-green-600">
                              <CheckCircle className="h-4 w-4" />
                              <span className="text-sm">Recorded (confidence: {(questionState.confidence * 100).toFixed(0)}%)</span>
                            </div>
                          )}
                        </div>
                        
                        {questionState.transcript && questionState.transcript.trim() && (
                          <div className="bg-green-50 border border-green-200 p-3 rounded">
                            <p className="text-sm font-medium text-green-700">✅ Transcribed Answer for Question {index + 1}:</p>
                            <p className="text-gray-900 mt-1 font-medium">{questionState.transcript}</p>
                            <p className="text-xs text-green-600 mt-1">Confidence: {(questionState.confidence * 100).toFixed(0)}%</p>
                          </div>
                        )}
                        
                        {!questionState.transcript && !questionState.isRecording && !questionState.error && (
                          <div className="bg-yellow-50 border border-yellow-200 p-2 rounded">
                            <p className="text-yellow-800 text-sm">⏺️ Click "Record Answer" to provide your voice response</p>
                          </div>
                        )}
                        
                        {questionState.error && (
                          <div className="bg-red-50 border border-red-200 rounded p-2">
                            <p className="text-red-800 text-sm">{questionState.error}</p>
                          </div>
                        )}
                        
                        {/* Hidden input for form submission */}
                        <Input
                          type="hidden"
                          value={answers[index] || ''}
                          onChange={(e) => handleAnswerChange(index, e.target.value)}
                        />
                      </div>
                    );
                  })}

                  {error && (
                    <div className="bg-red-50 border border-red-200 rounded p-3">
                      <p className="text-red-800 text-sm">{error}</p>
                    </div>
                  )}

                  <div className="flex gap-3">
                    <Button
                      type="submit"
                      disabled={isSubmitting || timeRemaining <= 0 || aiQuestions.length === 0 || !servicesInitialized}
                      className="flex-1 bg-blue-600 hover:bg-blue-700"
                    >
                      {isSubmitting ? (
                        <div className="flex items-center">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                          Verifying...
                        </div>
                      ) : (
                        <div className="flex items-center">
                          <Lock className="mr-2 h-4 w-4" />
                          Verify Identity
                        </div>
                      )}
                    </Button>
                    
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => {
                        localStorage.removeItem('authToken');
                        navigate('/login');
                      }}
                      className="px-6"
                    >
                      Logout
                    </Button>
                  </div>
                </form>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Additional Security Notice */}
        <Card className="border-gray-200 bg-gray-50">
          <CardContent className="pt-6">
            <div className="text-center text-sm text-gray-600">
              <p className="mb-2">
                <strong>Security Notice:</strong> If you did not perform these activities, please contact our support team immediately.
              </p>
              <p>
                Your session will be automatically terminated if verification is not completed within the time limit.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}