import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function TrainingEvalView({ data, mode = 'training' }) {
  // Checkbox states for toggling lines
  const [showTrainingLoss, setShowTrainingLoss] = useState(true);
  const [showTestingLoss, setShowTestingLoss] = useState(false);
  const [showValidationLoss, setShowValidationLoss] = useState(false);
  const [showTrainingAcc, setShowTrainingAcc] = useState(true);
  const [showTestingAcc, setShowTestingAcc] = useState(false);
  const [showValidationAcc, setShowValidationAcc] = useState(false);
  if (!data || data.error) {
    return (
      <div style={{ padding: 24, height: '100%', overflow: 'auto' }}>
        <h3 style={{ marginTop: 0, color: 'var(--text)' }}>{mode === 'training' ? 'Training Results' : 'Evaluation Results'}</h3>
        <div style={{ marginTop: 24, padding: 16, background: 'var(--panel)', borderRadius: 8, border: '1px solid var(--border)' }}>
          <div style={{ color: 'var(--muted)', fontSize: 14 }}>
            {data?.error ? `Error: ${data.error}` : 'Waiting for training data...'}
          </div>
          <div style={{ marginTop: 8, color: 'var(--muted)', fontSize: 12 }}>
            Run your model with VIZ=1 to export training metrics.
          </div>
        </div>
      </div>
    );
  }

  const { model_summary, total_params, trainable_params, best_loss, best_accuracy, num_epochs, current_epoch, is_training, epoch_losses, epoch_accuracies,
          train_loss, test_loss, train_accuracy, test_accuracy,
          training_loss, testing_loss, training_accuracy, testing_accuracy,
          validation_loss, val_loss, validation_accuracy, val_accuracy,
          epoch_validation_losses, epoch_val_losses, epoch_validation_accuracies, epoch_val_accuracies,
          epoch_training_losses, epoch_testing_losses, epoch_train_losses, epoch_test_losses,
          epoch_training_accuracies, epoch_testing_accuracies, epoch_train_accuracies, epoch_test_accuracies,
          early_stopped, expected_epochs, actual_epochs } = data;

  // Get separate training/testing arrays if available, otherwise use combined arrays
  const trainingLosses = epoch_training_losses || epoch_train_losses || epoch_losses || [];
  const testingLosses = epoch_testing_losses || epoch_test_losses || epoch_losses || [];
  const trainingAccuracies = epoch_training_accuracies || epoch_train_accuracies || epoch_accuracies || [];
  const testingAccuracies = epoch_testing_accuracies || epoch_test_accuracies || epoch_accuracies || [];

  // Filter out null/undefined values (epochs not yet completed), but keep 0 values
  const completedTrainingLosses = trainingLosses.filter(l => l != null && l >= 0);
  const completedTestingLosses = testingLosses.filter(l => l != null && l >= 0);
  const completedTrainingAccuracies = trainingAccuracies.filter(a => a != null && a >= 0);
  const completedTestingAccuracies = testingAccuracies.filter(a => a != null && a >= 0);
  const completedEpochs = Math.max(
    completedTrainingLosses.length,
    completedTestingLosses.length,
    completedTrainingAccuracies.length,
    completedTestingAccuracies.length
  );

  // Get validation data if available
  const validationLosses = epoch_validation_losses || epoch_val_losses || [];
  const completedValidationLosses = validationLosses.filter(l => l != null && l >= 0);
  const validationAccuracies = epoch_validation_accuracies || epoch_val_accuracies || [];
  const completedValidationAccuracies = validationAccuracies.filter(a => a != null && a >= 0);

  // Check if testing data exists (non-zero values only, since test is often evaluated at the end)
  const hasTestingData = !!(
    (epoch_testing_losses && epoch_testing_losses.some(v => v > 0)) ||
    (epoch_test_losses && epoch_test_losses.some(v => v > 0)) ||
    (epoch_testing_accuracies && epoch_testing_accuracies.some(v => v > 0)) ||
    (epoch_test_accuracies && epoch_test_accuracies.some(v => v > 0)) ||
    (testing_loss && testing_loss > 0) || (test_loss && test_loss > 0) ||
    (testing_accuracy && testing_accuracy > 0) || (test_accuracy && test_accuracy > 0)
  );

  // Check if validation data exists (after arrays are processed)
  // Check if arrays exist and have any non-zero values (validation is evaluated periodically)
  const hasValidationData = !!(validation_loss || val_loss ||
                                (epoch_validation_losses && epoch_validation_losses.length > 0 && epoch_validation_losses.some(v => v > 0)) ||
                                (epoch_val_losses && epoch_val_losses.length > 0 && epoch_val_losses.some(v => v > 0)) ||
                                validation_accuracy || val_accuracy ||
                                (epoch_validation_accuracies && epoch_validation_accuracies.length > 0 && epoch_validation_accuracies.some(v => v > 0)) ||
                                (epoch_val_accuracies && epoch_val_accuracies.length > 0 && epoch_val_accuracies.some(v => v > 0)) ||
                                completedValidationLosses.length > 0 || completedValidationAccuracies.length > 0);

  // Only show checkboxes if there are multiple data sources (testing or validation)
  const showCheckboxes = hasTestingData || hasValidationData;

  // Get latest training and testing metrics
  // Try different field names for compatibility
  const latestTrainingLoss = training_loss ?? train_loss ?? (completedTrainingLosses.length > 0 ? completedTrainingLosses[completedTrainingLosses.length - 1] : 0);
  const latestTestingLoss = (() => {
    if (testing_loss ?? false) return testing_loss;
    if (test_loss ?? false) return test_loss;
    for (let i = completedTestingLosses.length - 1; i >= 0; i--) {
      if (completedTestingLosses[i] > 0) return completedTestingLosses[i];
    }
    return null;
  })();
  const latestValidationLoss = validation_loss ?? val_loss ?? (completedValidationLosses.length > 0 ? completedValidationLosses[completedValidationLosses.length - 1] : null);
  const latestTrainingAccuracy = training_accuracy ?? train_accuracy ?? (completedTrainingAccuracies.length > 0 ? completedTrainingAccuracies[completedTrainingAccuracies.length - 1] : 0);
  const latestTestingAccuracy = (() => {
    if (testing_accuracy ?? false) return testing_accuracy;
    if (test_accuracy ?? false) return test_accuracy;
    for (let i = completedTestingAccuracies.length - 1; i >= 0; i--) {
      if (completedTestingAccuracies[i] > 0) return completedTestingAccuracies[i];
    }
    return null;
  })();
  const latestValidationAccuracy = validation_accuracy ?? val_accuracy ?? (completedValidationAccuracies.length > 0 ? completedValidationAccuracies[completedValidationAccuracies.length - 1] : null);

  // Learning rate data (needed for chart data creation)
  const learningRate = (data.learning_rate !== undefined && data.learning_rate !== null) ? data.learning_rate :
                       ((data.lr !== undefined && data.lr !== null) ? data.lr :
                       ((data.current_lr !== undefined && data.current_lr !== null) ? data.current_lr : null));
  const lrSchedule = data.lr_schedule || data.learning_rate_schedule || data.scheduler || null;
  const lrScheduleParams = data.lr_schedule_params || null;
  const lrHistory = data.epoch_learning_rates || data.lr_history || data.learning_rate_history || null;

  // Prepare data for Recharts with all metrics
  // Use actual completed epochs for dynamic x-axis (supports early stopping)
  // When early stopped, use actual_epochs; otherwise use completedEpochs or num_epochs
  const effectiveMaxEpochs = early_stopped && actual_epochs
    ? actual_epochs
    : Math.max(
        completedEpochs,
        completedValidationLosses.length,
        completedValidationAccuracies.length,
        completedTestingLosses.length,
        completedTestingAccuracies.length,
        num_epochs || completedEpochs
      );

  // Create chart data array - only include actual data (dynamic x-axis)
  const chartData = Array.from({ length: effectiveMaxEpochs }, (_, i) => {
    // Get test metrics from arrays first, then fall back to single float values at the last epoch
    let testLoss = null;
    let testAcc = null;

    if (i < completedTestingLosses.length && completedTestingLosses[i] > 0) {
      testLoss = completedTestingLosses[i];
    } else if (i === effectiveMaxEpochs - 1 && (testing_loss || test_loss)) {
      // Add test metrics at the last epoch if they exist as single float values
      testLoss = testing_loss || test_loss;
    }

    if (i < completedTestingAccuracies.length && completedTestingAccuracies[i] > 0) {
      testAcc = completedTestingAccuracies[i];
    } else if (i === effectiveMaxEpochs - 1 && (testing_accuracy || test_accuracy)) {
      // Add test metrics at the last epoch if they exist as single float values
      testAcc = testing_accuracy || test_accuracy;
    }

    // Get learning rate for this epoch
    let epochLR = null;
    if (lrHistory && i < lrHistory.length && lrHistory[i] > 0) {
      epochLR = lrHistory[i];
    } else if (learningRate && (!lrHistory || i >= lrHistory.length)) {
      // If no history but we have a current LR, use it for all epochs (constant LR)
      epochLR = learningRate;
    }

    return {
      epoch: i + 1,
      trainingLoss: completedTrainingLosses[i] != null ? completedTrainingLosses[i] : null,
      testingLoss: testLoss,
      validationLoss: completedValidationLosses[i] != null ? completedValidationLosses[i] : null,
      trainingAccuracy: completedTrainingAccuracies[i] != null ? completedTrainingAccuracies[i] * 100 : null,
      testingAccuracy: testAcc != null ? testAcc * 100 : null,
      validationAccuracy: completedValidationAccuracies[i] != null ? completedValidationAccuracies[i] * 100 : null,
      learningRate: epochLR
    };
  });

  // Show training status with early stopping support
  const effectiveNumEpochs = num_epochs || completedEpochs;
  const effectiveExpectedEpochs = expected_epochs || effectiveNumEpochs;
  const effectiveActualEpochs = actual_epochs || completedEpochs;

  let trainingStatus = '';
  if (is_training) {
    trainingStatus = `Training... (${current_epoch || completedEpochs}/${effectiveNumEpochs})`;
  } else if (early_stopped) {
    trainingStatus = `Early Stopped (${effectiveActualEpochs}/${effectiveExpectedEpochs})`;
  } else {
    trainingStatus = `Completed (${completedEpochs}/${effectiveNumEpochs})`;
  }

  // Calculate training metrics with early stopping support
  const currentEpochNum = current_epoch || completedEpochs || 0;
  const totalEpochs = early_stopped ? effectiveActualEpochs : (num_epochs || 1);
  const epochProgress = totalEpochs > 0 ? (currentEpochNum / totalEpochs) * 100 : 0;
  const overallProgress = totalEpochs > 0 ? (completedEpochs / totalEpochs) * 100 : 0;

  // Loss trend analysis
  const getLossTrend = () => {
    if (completedTrainingLosses.length < 3) return 'insufficient';
    const recent = completedTrainingLosses.slice(-5);
    const older = completedTrainingLosses.slice(-10, -5);
    if (older.length === 0) return 'insufficient';
    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;
    const change = ((olderAvg - recentAvg) / olderAvg) * 100;
    if (change > 1) return 'improving';
    if (change < -1) return 'degrading';
    return 'plateau';
  };

  const lossTrend = getLossTrend();
  const lossTrendColor = {
    improving: '#10b981',
    degrading: '#ef4444',
    plateau: '#f59e0b',
    insufficient: '#6b7280'
  };

  // Loss reduction rate (% change per epoch)
  // Loss reduction rate (use exported value if available, otherwise calculate)
  const getLossReductionRate = () => {
    if (data.loss_reduction_rate !== undefined && data.loss_reduction_rate !== null) {
      return data.loss_reduction_rate;
    }
    if (completedTrainingLosses.length < 2) return null;
    const last = completedTrainingLosses[completedTrainingLosses.length - 1];
    const prev = completedTrainingLosses[completedTrainingLosses.length - 2];
    if (prev === 0) return null;
    return ((prev - last) / prev) * 100;
  };

  const lossReductionRate = getLossReductionRate();

  // Best loss and epoch
  const bestLossEpoch = completedTrainingLosses.indexOf(best_loss) + 1;

  // Loss stability (variance over recent epochs)
  // Loss stability (use exported value if available, otherwise calculate)
  const getLossStability = () => {
    if (data.loss_stability !== undefined && data.loss_stability !== null) {
      return data.loss_stability;
    }
    if (completedTrainingLosses.length < 5) return null;
    const recent = completedTrainingLosses.slice(-10);
    const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
    const variance = recent.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / recent.length;
    return Math.sqrt(variance);
  };

  const lossStability = getLossStability();

  // Time metrics (check for 0 explicitly, as 0 is a valid value)
  const epochTime = (data.epoch_time !== undefined && data.epoch_time !== null) ? data.epoch_time :
                    ((data.time_per_epoch !== undefined && data.time_per_epoch !== null) ? data.time_per_epoch : null);
  const totalTime = (data.total_time !== undefined && data.total_time !== null) ? data.total_time :
                    ((data.elapsed_time !== undefined && data.elapsed_time !== null) ? data.elapsed_time : null);
  const avgEpochTime = (epochTime !== null && epochTime > 0) ? epochTime :
                       ((totalTime !== null && totalTime > 0 && completedEpochs > 0) ? totalTime / completedEpochs : null);
  const estimatedRemaining = (data.estimated_remaining !== undefined && data.estimated_remaining !== null && data.estimated_remaining > 0)
    ? data.estimated_remaining
    : (avgEpochTime && totalEpochs > currentEpochNum ? avgEpochTime * (totalEpochs - currentEpochNum) : null);

  // Throughput
  const throughput = data.throughput || data.samples_per_sec || data.tokens_per_sec || null;
  const epochsPerHour = (data.epochs_per_hour !== undefined && data.epochs_per_hour !== null && data.epochs_per_hour > 0)
    ? data.epochs_per_hour
    : (avgEpochTime && avgEpochTime > 0 ? 3600 / avgEpochTime : null);


  // Gradient health (check for 0 explicitly)
  const gradientNorm = (data.gradient_norm !== undefined && data.gradient_norm !== null) ? data.gradient_norm :
                       ((data.grad_norm !== undefined && data.grad_norm !== null) ? data.grad_norm :
                       ((data.gradient_norm_avg !== undefined && data.gradient_norm_avg !== null) ? data.gradient_norm_avg : null));
  const getGradientHealth = () => {
    if (!gradientNorm) return null;
    if (gradientNorm > 100) return { status: 'exploding', color: '#ef4444', text: 'Exploding' };
    if (gradientNorm < 0.001) return { status: 'vanishing', color: '#ef4444', text: 'Vanishing' };
    if (gradientNorm < 0.01) return { status: 'low', color: '#f59e0b', text: 'Low' };
    return { status: 'healthy', color: '#10b981', text: 'Healthy' };
  };
  const gradientHealth = getGradientHealth();

  return (
    <div style={{ padding: 12, height: '100%', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, flexShrink: 0 }}>
        <h3 style={{ margin: 0, color: 'var(--text)', fontSize: 16 }}>{mode === 'training' ? 'Training Results' : 'Evaluation Results'}</h3>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <div style={{
            padding: '6px 14px',
            background: is_training
              ? 'linear-gradient(135deg, #4a90e2, #6366f1)'
              : early_stopped
              ? 'linear-gradient(135deg, #f59e0b, #d97706)'
              : 'linear-gradient(135deg, #10b981, #059669)',
            color: 'white',
            borderRadius: 8,
            fontSize: 11,
            fontWeight: 'bold',
            boxShadow: is_training
              ? '0 2px 8px rgba(74, 144, 226, 0.3)'
              : early_stopped
              ? '0 2px 8px rgba(245, 158, 11, 0.3)'
              : '0 2px 8px rgba(16, 185, 129, 0.3)',
            transition: 'all 0.2s ease'
          }}>
            {trainingStatus}
          </div>
          {early_stopped && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '4px 10px',
              borderRadius: 4,
              fontSize: 10,
              fontWeight: 600,
              background: 'rgba(245, 158, 11, 0.1)',
              color: '#f59e0b',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
              </svg>
              <span>Early Stop</span>
            </div>
          )}
        </div>
      </div>

      {/* Training, Testing, and Validation Metrics */}
      {(() => {
        const cardCount = 2 + (hasTestingData ? 2 : 0) + (hasValidationData ? 2 : 0);
        const gridCols = `repeat(${cardCount}, 1fr)`;
        return (
      <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: gridCols, gap: 12, flexShrink: 0 }}>
        <div className="card" style={{
          padding: 14,
          background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(99, 102, 241, 0.05))',
          borderRadius: 10,
          border: '1px solid rgba(99, 102, 241, 0.2)',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
        }}>
          <div style={{ color: 'var(--muted)', fontSize: 11, fontWeight: 500, marginBottom: 6 }}>Training Loss</div>
          <div style={{ color: 'var(--text)', fontSize: 20, fontWeight: 'bold', fontFamily: 'monospace' }}>{latestTrainingLoss?.toFixed(6) || 'N/A'}</div>
        </div>
        {hasTestingData && (
        <div className="card" style={{
          padding: 14,
          background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05))',
          borderRadius: 10,
          border: '1px solid rgba(245, 158, 11, 0.2)',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
        }}>
          <div style={{ color: 'var(--muted)', fontSize: 11, fontWeight: 500, marginBottom: 6 }}>Testing Loss</div>
          <div style={{ color: 'var(--text)', fontSize: 20, fontWeight: 'bold', fontFamily: 'monospace' }}>{latestTestingLoss != null ? latestTestingLoss.toFixed(6) : 'N/A'}</div>
        </div>
        )}
        {hasValidationData && (
          <div className="card" style={{
            padding: 14,
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05))',
            borderRadius: 10,
            border: '1px solid rgba(239, 68, 68, 0.2)',
            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
          }}>
            <div style={{ color: 'var(--muted)', fontSize: 11, fontWeight: 500, marginBottom: 6 }}>Validation Loss</div>
            <div style={{ color: 'var(--text)', fontSize: 20, fontWeight: 'bold', fontFamily: 'monospace' }}>{latestValidationLoss?.toFixed(6) || 'N/A'}</div>
          </div>
        )}
        <div className="card" style={{
          padding: 14,
          background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05))',
          borderRadius: 10,
          border: '1px solid rgba(16, 185, 129, 0.2)',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
        }}>
          <div style={{ color: 'var(--muted)', fontSize: 11, fontWeight: 500, marginBottom: 6 }}>Training Accuracy</div>
          <div style={{ color: 'var(--text)', fontSize: 20, fontWeight: 'bold', fontFamily: 'monospace' }}>{(latestTrainingAccuracy * 100)?.toFixed(2) || 'N/A'}%</div>
        </div>
        {hasTestingData && (
        <div className="card" style={{
          padding: 14,
          background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05))',
          borderRadius: 10,
          border: '1px solid rgba(245, 158, 11, 0.2)',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
        }}>
          <div style={{ color: 'var(--muted)', fontSize: 11, fontWeight: 500, marginBottom: 6 }}>Testing Accuracy</div>
          <div style={{ color: 'var(--text)', fontSize: 20, fontWeight: 'bold', fontFamily: 'monospace' }}>{latestTestingAccuracy != null ? (latestTestingAccuracy * 100).toFixed(2) : 'N/A'}%</div>
        </div>
        )}
        {hasValidationData && (
          <div className="card" style={{
            padding: 14,
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05))',
            borderRadius: 10,
            border: '1px solid rgba(239, 68, 68, 0.2)',
            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
          }}>
            <div style={{ color: 'var(--muted)', fontSize: 11, fontWeight: 500, marginBottom: 6 }}>Validation Accuracy</div>
            <div style={{ color: 'var(--text)', fontSize: 20, fontWeight: 'bold', fontFamily: 'monospace' }}>{(latestValidationAccuracy * 100)?.toFixed(2) || 'N/A'}%</div>
          </div>
        )}
      </div>
        );
      })()}

      {/* Charts Side by Side */}
      <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, flex: '1 1 auto', minHeight: 320 }}>
        {/* Loss Curve */}
        <div className="card" style={{
          padding: 14,
          background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(99, 102, 241, 0.02))',
          borderRadius: 10,
          border: '1px solid rgba(99, 102, 241, 0.15)',
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexShrink: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h4 style={{ margin: 0, color: 'var(--text)', fontSize: 14, fontWeight: 600 }}>Loss Curve</h4>
              {early_stopped && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  padding: '2px 6px',
                  borderRadius: 3,
                  fontSize: 9,
                  fontWeight: 600,
                  background: 'rgba(245, 158, 11, 0.15)',
                  color: '#f59e0b',
                  border: '1px solid rgba(245, 158, 11, 0.3)'
                }}>
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                  </svg>
                  <span>Early Stop</span>
                </div>
              )}
            </div>
            {showCheckboxes && (
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 11, color: 'var(--text)' }}>
                  <input
                    type="checkbox"
                    checked={showTrainingLoss}
                    onChange={(e) => setShowTrainingLoss(e.target.checked)}
                    style={{ cursor: 'pointer', accentColor: '#4a90e2' }}
                  />
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ width: 12, height: 2, background: '#4a90e2', borderRadius: 1 }}></span>
                    <span>Training</span>
                  </span>
                </label>
                {hasTestingData && (
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 11, color: 'var(--text)' }}>
                    <input
                      type="checkbox"
                      checked={showTestingLoss}
                      onChange={(e) => setShowTestingLoss(e.target.checked)}
                      style={{ cursor: 'pointer', accentColor: '#f59e0b' }}
                    />
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 12, height: 2, background: '#f59e0b', borderRadius: 1 }}></span>
                      <span>Testing</span>
                    </span>
                  </label>
                )}
                {hasValidationData && (
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 11, color: 'var(--text)' }}>
                    <input
                      type="checkbox"
                      checked={showValidationLoss}
                      onChange={(e) => setShowValidationLoss(e.target.checked)}
                      style={{ cursor: 'pointer', accentColor: '#ef4444' }}
                    />
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 12, height: 2, background: '#ef4444', borderRadius: 1 }}></span>
                      <span>Validation</span>
                    </span>
                  </label>
                )}
              </div>
            )}
          </div>
          <div style={{ height: 280, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                <XAxis
                  dataKey="epoch"
                  stroke="#6b7280"
                  tick={{ fill: '#6b7280', fontSize: 9 }}
                  style={{ fontFamily: 'monospace' }}
                  type="number"
                  domain={[1, 'dataMax']}
                  allowDecimals={false}
                />
                <YAxis
                  stroke="#6b7280"
                  tick={{ fill: '#6b7280', fontSize: 9 }}
                  style={{ fontFamily: 'monospace' }}
                  domain={['auto', 'auto']}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0, 0, 0, 0.9)',
                    border: '1px solid #4a90e2',
                    borderRadius: '4px',
                    color: 'white',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                    padding: '6px 10px'
                  }}
                  formatter={(value, name) => [value !== null ? value.toFixed(6) : 'N/A', name]}
                  labelFormatter={(label) => `Epoch: ${label}`}
                />
                {showTrainingLoss && (
                  <Line
                    type="monotone"
                    dataKey="trainingLoss"
                    stroke="#4a90e2"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5, fill: '#4a90e2', stroke: 'white', strokeWidth: 2 }}
                    name="Training Loss"
                    connectNulls={false}
                  />
                )}
                {showTestingLoss && hasTestingData && (
                  <Line
                    type="monotone"
                    dataKey="testingLoss"
                    stroke="#f59e0b"
                    strokeWidth={2.5}
                    dot={{ r: 5, fill: '#f59e0b', stroke: '#111827', strokeWidth: 1 }}
                    activeDot={{ r: 6, fill: '#f59e0b', stroke: 'white', strokeWidth: 2 }}
                    name="Testing Loss"
                    connectNulls={false}
                    strokeDasharray="4 4"
                  />
                )}
                {showValidationLoss && hasValidationData && (
                  <Line
                    type="monotone"
                    dataKey="validationLoss"
                    stroke="#ef4444"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5, fill: '#ef4444', stroke: 'white', strokeWidth: 2 }}
                    name="Validation Loss"
                    connectNulls={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Accuracy Curve */}
        <div className="card" style={{
          padding: 14,
          background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(16, 185, 129, 0.02))',
          borderRadius: 10,
          border: '1px solid rgba(16, 185, 129, 0.15)',
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexShrink: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h4 style={{ margin: 0, color: 'var(--text)', fontSize: 14, fontWeight: 600 }}>Accuracy Curve</h4>
              {early_stopped && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  padding: '2px 6px',
                  borderRadius: 3,
                  fontSize: 9,
                  fontWeight: 600,
                  background: 'rgba(245, 158, 11, 0.15)',
                  color: '#f59e0b',
                  border: '1px solid rgba(245, 158, 11, 0.3)'
                }}>
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                  </svg>
                  <span>Early Stop</span>
                </div>
              )}
            </div>
            {showCheckboxes && (
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 11, color: 'var(--text)' }}>
                  <input
                    type="checkbox"
                    checked={showTrainingAcc}
                    onChange={(e) => setShowTrainingAcc(e.target.checked)}
                    style={{ cursor: 'pointer', accentColor: '#10b981' }}
                  />
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ width: 12, height: 2, background: '#10b981', borderRadius: 1 }}></span>
                    <span>Training</span>
                  </span>
                </label>
                {hasTestingData && (
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 11, color: 'var(--text)' }}>
                    <input
                      type="checkbox"
                      checked={showTestingAcc}
                      onChange={(e) => setShowTestingAcc(e.target.checked)}
                      style={{ cursor: 'pointer', accentColor: '#f59e0b' }}
                    />
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 12, height: 2, background: '#f59e0b', borderRadius: 1 }}></span>
                      <span>Testing</span>
                    </span>
                  </label>
                )}
                {hasValidationData && (
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 11, color: 'var(--text)' }}>
                    <input
                      type="checkbox"
                      checked={showValidationAcc}
                      onChange={(e) => setShowValidationAcc(e.target.checked)}
                      style={{ cursor: 'pointer', accentColor: '#ef4444' }}
                    />
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 12, height: 2, background: '#ef4444', borderRadius: 1 }}></span>
                      <span>Validation</span>
                    </span>
                  </label>
                )}
              </div>
            )}
          </div>
          <div style={{ height: 280, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                <XAxis
                  dataKey="epoch"
                  stroke="#6b7280"
                  tick={{ fill: '#6b7280', fontSize: 9 }}
                  style={{ fontFamily: 'monospace' }}
                  type="number"
                  domain={[1, 'dataMax']}
                  allowDecimals={false}
                />
                <YAxis
                  stroke="#6b7280"
                  tick={{ fill: '#6b7280', fontSize: 9 }}
                  style={{ fontFamily: 'monospace' }}
                  domain={['auto', 'auto']}
                  tickFormatter={(value) => `${value.toFixed(1)}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0, 0, 0, 0.9)',
                    border: '1px solid #10b981',
                    borderRadius: '4px',
                    color: 'white',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                    padding: '6px 10px'
                  }}
                  formatter={(value, name) => [value !== null ? `${value.toFixed(2)}%` : 'N/A', name]}
                  labelFormatter={(label) => `Epoch: ${label}`}
                />
                {showTrainingAcc && (
                  <Line
                    type="monotone"
                    dataKey="trainingAccuracy"
                    stroke="#10b981"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5, fill: '#10b981', stroke: 'white', strokeWidth: 2 }}
                    name="Training Accuracy"
                    connectNulls={false}
                  />
                )}
                {showTestingAcc && hasTestingData && (
                  <Line
                    type="monotone"
                    dataKey="testingAccuracy"
                    stroke="#f59e0b"
                    strokeWidth={2.5}
                    dot={{ r: 5, fill: '#f59e0b', stroke: '#111827', strokeWidth: 1 }}
                    activeDot={{ r: 6, fill: '#f59e0b', stroke: 'white', strokeWidth: 2 }}
                    name="Testing Accuracy"
                    connectNulls={false}
                    strokeDasharray="4 4"
                  />
                )}
                {showValidationAcc && hasValidationData && (
                  <Line
                    type="monotone"
                    dataKey="validationAccuracy"
                    stroke="#ef4444"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5, fill: '#ef4444', stroke: 'white', strokeWidth: 2 }}
                    name="Validation Accuracy"
                    connectNulls={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Training Metrics and Epoch Table Side by Side */}
      <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, flex: '1 1 auto', minHeight: 0 }}>
        {/* Training Metrics Panel */}
        <div className="card" style={{
          padding: 16,
          background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05))',
          borderRadius: 10,
          border: '1px solid rgba(16, 185, 129, 0.2)',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          height: '100%',
          minHeight: 0,
          overflow: 'hidden'
        }}>
          <h4 style={{
            margin: '0 0 12px 0',
            color: 'var(--text)',
            fontSize: 14,
            fontWeight: 600,
            flexShrink: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 8
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{
                width: 3,
                height: 16,
                background: 'linear-gradient(135deg, #10b981, #059669)',
                borderRadius: 2
              }}></span>
              Training Metrics
            </div>
            {/* Convergence Status Icon */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '4px 10px',
              background: lossTrend === 'improving'
                ? 'rgba(16, 185, 129, 0.15)'
                : lossTrend === 'degrading'
                ? 'rgba(239, 68, 68, 0.15)'
                : 'rgba(245, 158, 11, 0.15)',
              borderRadius: 12,
              border: `1px solid ${lossTrendColor[lossTrend]}60`,
              cursor: 'default',
              position: 'relative'
            }}
            title={lossTrend === 'improving' ? 'Converging - Loss is improving' :
                   lossTrend === 'degrading' ? 'Diverging - Loss is increasing' :
                   lossTrend === 'plateau' ? 'Plateaued - Loss has stabilized' : 'Initializing - Not enough data'}
            >
              <div style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: lossTrendColor[lossTrend],
                boxShadow: `0 0 6px ${lossTrendColor[lossTrend]}80`
              }}></div>
              <span style={{
                color: 'var(--text)',
                fontSize: 10,
                fontWeight: 600
              }}>
                {lossTrend === 'improving' ? '↓' :
                 lossTrend === 'degrading' ? '↑' :
                 lossTrend === 'plateau' ? '→' : '—'}
              </span>
              <span style={{
                color: 'var(--text)',
                fontSize: 10,
                fontWeight: 500,
                marginLeft: 2
              }}>
                {lossTrend === 'improving' ? 'Converging' :
                 lossTrend === 'degrading' ? 'Diverging' :
                 lossTrend === 'plateau' ? 'Plateau' : 'Init'}
              </span>
            </div>
          </h4>

          <div style={{ flex: '1 1 auto', minHeight: 0, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
          {/* Progress Bar */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span style={{ color: 'var(--muted)', fontSize: 11, fontWeight: 500 }}>Epoch Progress</span>
              <span style={{ color: 'var(--text)', fontSize: 11, fontFamily: 'monospace' }}>{currentEpochNum}/{totalEpochs}</span>
            </div>
            <div style={{
              width: '100%',
              height: 8,
              background: 'rgba(0, 0, 0, 0.2)',
              borderRadius: 4,
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${epochProgress}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #10b981, #059669)',
                transition: 'width 0.3s ease',
                boxShadow: '0 0 8px rgba(16, 185, 129, 0.5)'
              }}></div>
            </div>
          </div>

          {/* Combined Metrics */}
          <div style={{ marginBottom: 16, padding: 12, background: 'rgba(0, 0, 0, 0.2)', borderRadius: 6 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div>
                <div style={{ color: 'var(--muted)', fontSize: 10, marginBottom: 4 }}>Time/Epoch</div>
                <div style={{ color: 'var(--text)', fontSize: 12, fontFamily: 'monospace', fontWeight: 600 }}>
                  {avgEpochTime ? (() => {
                    if (avgEpochTime >= 3600) {
                      return `${(avgEpochTime / 3600).toFixed(2)}hr`;
                    } else if (avgEpochTime >= 60) {
                      return `${(avgEpochTime / 60).toFixed(2)}min`;
                    } else {
                      return `${avgEpochTime.toFixed(2)}s`;
                    }
                  })() : 'N/A'}
                </div>
              </div>
              <div>
                <div style={{ color: 'var(--muted)', fontSize: 10, marginBottom: 4 }}>Total Time</div>
                <div style={{ color: 'var(--text)', fontSize: 12, fontFamily: 'monospace', fontWeight: 600 }}>
                  {totalTime ? `${(totalTime / 60).toFixed(1)}m` : 'N/A'}
                </div>
              </div>
              <div>
                <div style={{ color: 'var(--muted)', fontSize: 10, marginBottom: 4 }}>Est. Remaining</div>
                <div style={{ color: 'var(--text)', fontSize: 12, fontFamily: 'monospace', fontWeight: 600 }}>
                  {estimatedRemaining ? `${(estimatedRemaining / 60).toFixed(1)}m` : 'N/A'}
                </div>
              </div>
              <div>
                <div style={{ color: 'var(--muted)', fontSize: 10, marginBottom: 4 }}>Epochs/Hour</div>
                <div style={{ color: 'var(--text)', fontSize: 12, fontFamily: 'monospace', fontWeight: 600 }}>
                  {epochsPerHour ? (() => {
                    if (epochsPerHour >= 1) {
                      return `${epochsPerHour.toFixed(1)}/hr`;
                    } else if (epochsPerHour * 60 >= 1) {
                      return `${(epochsPerHour * 60).toFixed(1)}/min`;
                    } else {
                      return `${(epochsPerHour * 3600).toFixed(1)}/sec`;
                    }
                  })() : 'N/A'}
                </div>
              </div>
              <div>
                <div style={{ color: 'var(--muted)', fontSize: 10, marginBottom: 4 }}>Learning Rate</div>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  fontSize: 12,
                  fontFamily: 'monospace',
                  fontWeight: 600
                }}>
                  {learningRate ? (
                    <>
                      <span>{learningRate < 0.001 ? learningRate.toExponential(2) : learningRate.toFixed(6)}</span>
                      {lrSchedule && (
                        <span style={{
                          fontSize: 9,
                          color: 'var(--muted)',
                          marginLeft: 4,
                          padding: '2px 6px',
                          background: 'rgba(99, 102, 241, 0.15)',
                          borderRadius: 4,
                          border: '1px solid rgba(99, 102, 241, 0.2)'
                        }} title={lrScheduleParams ? `Scheduler: ${lrSchedule} (${lrScheduleParams})` : `Scheduler: ${lrSchedule}`}>
                          {lrSchedule}
                          {lrScheduleParams && (
                            <span style={{ marginLeft: 3, fontSize: 8, opacity: 0.8 }}>
                              ({lrScheduleParams})
                            </span>
                          )}
                        </span>
                      )}
                      {!lrSchedule && (
                        <span style={{
                          fontSize: 9,
                          color: 'var(--muted)',
                          marginLeft: 4,
                          padding: '2px 6px',
                          background: 'rgba(107, 114, 128, 0.15)',
                          borderRadius: 4,
                          border: '1px solid rgba(107, 114, 128, 0.2)'
                        }} title="Constant learning rate">
                          Constant
                        </span>
                      )}
                    </>
                  ) : 'N/A'}
                </div>
              </div>
              <div>
                <div style={{ color: 'var(--muted)', fontSize: 10, marginBottom: 4 }}>Gradient Health</div>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  fontSize: 12,
                  fontWeight: 600,
                  color: gradientHealth?.color || 'var(--text)'
                }}>
                  {gradientHealth ? (
                    <>
                      <span style={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: gradientHealth.color,
                        boxShadow: `0 0 6px ${gradientHealth.color}80`
                      }}></span>
                      <span>{gradientHealth.text}</span>
                      {gradientNorm && (
                        <span style={{
                          fontSize: 10,
                          color: 'var(--muted)',
                          fontFamily: 'monospace',
                          marginLeft: 4
                        }}>
                          ({gradientNorm.toFixed(4)})
                        </span>
                      )}
                    </>
                  ) : (
                    <span>N/A</span>
                  )}
                </div>
              </div>
              <div>
                <div style={{ color: 'var(--muted)', fontSize: 10, marginBottom: 4 }}>Reduction Rate</div>
                <div style={{ color: 'var(--text)', fontSize: 12, fontFamily: 'monospace', fontWeight: 600 }}>
                  {lossReductionRate !== null ? `${lossReductionRate.toFixed(2)}%` : 'N/A'}
                </div>
              </div>
              <div>
                <div style={{ color: 'var(--muted)', fontSize: 10, marginBottom: 4 }}>Loss Stability (σ)</div>
                <div style={{ color: 'var(--text)', fontSize: 12, fontFamily: 'monospace', fontWeight: 600 }}>
                  {lossStability !== null ? lossStability.toFixed(6) : 'N/A'}
                </div>
              </div>
            </div>
            {throughput && (
              <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid rgba(16, 185, 129, 0.1)' }}>
                <div style={{ color: 'var(--muted)', fontSize: 10, marginBottom: 4 }}>Throughput</div>
                <div style={{ color: 'var(--text)', fontSize: 12, fontFamily: 'monospace', fontWeight: 600 }}>
                  {throughput.toLocaleString()} {data.tokens_per_sec ? 'tokens/s' : 'samples/s'}
                </div>
              </div>
            )}
          </div>
          {/* Add padding at bottom for scrolling */}
          <div style={{ height: 8 }}></div>
          </div>
        </div>

        {/* Epoch Table */}
        <div className="card" style={{
          padding: 16,
          background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(99, 102, 241, 0.05))',
          borderRadius: 10,
          border: '1px solid rgba(99, 102, 241, 0.2)',
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          minHeight: 0,
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          overflow: 'hidden'
        }}>
          <h4 style={{
            margin: '0 0 12px 0',
            color: 'var(--text)',
            fontSize: 14,
            fontWeight: 600,
            flexShrink: 0,
            display: 'flex',
            alignItems: 'center',
            gap: 8
          }}>
            <span style={{
              width: 3,
              height: 16,
              background: 'linear-gradient(135deg, #6366f1, #4a90e2)',
              borderRadius: 2
            }}></span>
            Epoch Summary
          </h4>
          <div style={{
            flex: '1 1 auto',
            minHeight: 0,
            overflow: 'auto',
            borderRadius: 6,
            background: 'rgba(0, 0, 0, 0.2)',
            border: '1px solid rgba(99, 102, 241, 0.1)',
            paddingBottom: 8
          }}>
            <table style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: 11
            }}>
              <thead style={{
                position: 'sticky',
                top: 0,
                background: 'rgba(99, 102, 241, 0.15)',
                backdropFilter: 'blur(10px)',
                zIndex: 1
              }}>
                <tr style={{ borderBottom: '2px solid rgba(99, 102, 241, 0.3)' }}>
                  <th style={{
                    padding: '10px 12px',
                    textAlign: 'left',
                    color: 'var(--text)',
                    fontSize: 11,
                    fontWeight: 600
                  }}>
                    Epoch
                  </th>
                  <th style={{
                    padding: '10px 12px',
                    textAlign: 'right',
                    color: 'var(--text)',
                    fontSize: 11,
                    fontWeight: 600
                  }}>
                    Loss
                  </th>
                  <th style={{
                    padding: '10px 12px',
                    textAlign: 'right',
                    color: 'var(--text)',
                    fontSize: 11,
                    fontWeight: 600
                  }}>
                    Accuracy
                  </th>
                </tr>
              </thead>
              <tbody>
                {completedTrainingLosses.map((loss, i) => {
                  const isLatest = i === completedTrainingLosses.length - 1;

                  return (
                    <tr
                      key={i}
                      style={{
                        borderBottom: '1px solid rgba(99, 102, 241, 0.1)',
                        background: isLatest ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
                        transition: 'all 0.15s ease',
                        cursor: 'pointer'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = 'rgba(99, 102, 241, 0.15)';
                        e.currentTarget.style.transform = 'translateX(2px)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = isLatest ? 'rgba(99, 102, 241, 0.1)' : 'transparent';
                        e.currentTarget.style.transform = 'translateX(0)';
                      }}
                    >
                      <td style={{
                        padding: '8px 12px',
                        color: isLatest ? 'var(--text)' : 'var(--text)',
                        fontSize: 11,
                        fontWeight: isLatest ? 600 : 400
                      }}>
                        {i + 1}
                        {isLatest && <span style={{
                          marginLeft: 6,
                          color: '#6366f1',
                          fontSize: 9
                        }}>●</span>}
                      </td>
                      <td style={{
                        padding: '8px 12px',
                        textAlign: 'right',
                        color: 'var(--text)',
                        fontFamily: 'monospace',
                        fontSize: 11,
                        fontWeight: isLatest ? 600 : 400
                      }}>
                        {loss.toFixed(6)}
                      </td>
                      <td style={{
                        padding: '8px 12px',
                        textAlign: 'right',
                        color: 'var(--text)',
                        fontFamily: 'monospace',
                        fontSize: 11,
                        fontWeight: isLatest ? 600 : 400
                      }}>
                        {(completedTrainingAccuracies[i] * 100).toFixed(2)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
