import React, { useState, useEffect } from 'react';
import { modelService, ModelStatus } from '../services/modelService';

interface ModelSelectorProps {
  onModelChange?: (model: string) => void;
  className?: string;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({ 
  onModelChange, 
  className = '' 
}) => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadModelStatus();
  }, []);

  const loadModelStatus = async () => {
    try {
      setLoading(true);
      setError(null);
      const status = await modelService.getModelStatus();
      setModelStatus(status);
    } catch (err) {
      setError('Failed to load model status');
      console.error('Error loading model status:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleModelChange = async (newModel: string) => {
    if (!modelStatus || newModel === modelStatus.current_model) return;

    try {
      setSwitching(true);
      setError(null);
      
      await modelService.switchModel(newModel);
      
      // Update local state
      setModelStatus(prev => prev ? { ...prev, current_model: newModel } : null);
      
      // Notify parent component
      onModelChange?.(newModel);
      
      // Show success message
      alert(`Model switched to ${modelService.getModelDisplayName(newModel)}. Please restart the backend for changes to take effect.`);
      
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to switch model');
      console.error('Error switching model:', err);
    } finally {
      setSwitching(false);
    }
  };

  if (loading) {
    return (
      <div className={`bg-white rounded-lg border border-gray-200 p-4 ${className}`}>
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Loading models...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-red-50 border border-red-200 rounded-lg p-4 ${className}`}>
        <div className="flex items-center">
          <div className="text-red-600 text-sm">
            <strong>Error:</strong> {error}
          </div>
          <button
            onClick={loadModelStatus}
            className="ml-4 text-red-600 hover:text-red-800 text-sm underline"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!modelStatus) {
    return (
      <div className={`bg-gray-50 border border-gray-200 rounded-lg p-4 ${className}`}>
        <div className="text-gray-600 text-sm">No model information available</div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg border border-gray-200 p-4 ${className}`}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">LLM Model Selection</h3>
        <div className="text-sm text-gray-600">
          Current: <span className="font-medium">{modelService.getModelDisplayName(modelStatus.current_model)}</span>
        </div>
        {modelStatus.memory_usage && (
          <div className="text-xs text-gray-500 mt-1">
            Memory Usage: {modelStatus.memory_usage}
          </div>
        )}
      </div>

      <div className="space-y-2">
        {modelStatus.available_models.map((model) => {
          const isCurrent = model === modelStatus.current_model;
          const requirements = modelService.getModelRequirements(model);
          
          return (
            <div
              key={model}
              className={`p-3 rounded-lg border cursor-pointer transition-all ${
                isCurrent
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
              } ${switching ? 'opacity-50 cursor-not-allowed' : ''}`}
              onClick={() => !switching && !isCurrent && handleModelChange(model)}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center">
                    <div className={`w-3 h-3 rounded-full mr-3 ${
                      isCurrent ? 'bg-blue-500' : 'bg-gray-300'
                    }`} />
                    <div>
                      <div className="font-medium text-gray-900">
                        {modelService.getModelDisplayName(model)}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        RAM: {requirements.ram} • Quality: {requirements.quality} • Speed: {requirements.speed}
                      </div>
                    </div>
                  </div>
                </div>
                <div className="ml-4">
                  {isCurrent ? (
                    <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                      Active
                    </span>
                  ) : (
                    <button
                      disabled={switching}
                      className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded hover:bg-gray-200 disabled:opacity-50"
                    >
                      {switching ? 'Switching...' : 'Switch'}
                    </button>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
        <div className="text-xs text-yellow-800">
          <strong>Note:</strong> After switching models, you need to restart the backend server for changes to take effect.
        </div>
      </div>
    </div>
  );
};
