import api from './api';

export interface ModelStatus {
  current_model: string;
  available_models: string[];
  status: string;
  memory_usage?: string;
}

export interface ModelSwitchResponse {
  message: string;
  current_model: string;
  restart_required: boolean;
}

export const modelService = {
  /**
   * Get current model status and available models
   */
  async getModelStatus(): Promise<ModelStatus> {
    try {
      const response = await api.get('/api/v1/admin/models');
      return response.data;
    } catch (error) {
      console.error('Failed to get model status:', error);
      throw error;
    }
  },

  /**
   * Switch to a different model
   */
  async switchModel(model: string): Promise<ModelSwitchResponse> {
    try {
      const response = await api.post('/api/v1/admin/models/switch', { model });
      return response.data;
    } catch (error) {
      console.error('Failed to switch model:', error);
      throw error;
    }
  },

  /**
   * Get model display name with size info
   */
  getModelDisplayName(model: string): string {
    const modelInfo: { [key: string]: string } = {
      'llama3.1:70b': 'Llama 3.1 70B (Maximum Quality)',
      'llama3.1:13b': 'Llama 3.1 13B (High Quality)',
      'llama3.1:8b': 'Llama 3.1 8B (Balanced)',
      'llama3.1:8b-instruct-q4_0': 'Llama 3.1 8B Q4 (Fast)',
    };
    return modelInfo[model] || model;
  },

  /**
   * Get model requirements
   */
  getModelRequirements(model: string): { ram: string; quality: string; speed: string } {
    const requirements: { [key: string]: { ram: string; quality: string; speed: string } } = {
      'llama3.1:70b': { ram: '42GB+', quality: 'Maximum', speed: 'Slow' },
      'llama3.1:13b': { ram: '16GB+', quality: 'High', speed: 'Medium' },
      'llama3.1:8b': { ram: '10GB+', quality: 'Good', speed: 'Fast' },
      'llama3.1:8b-instruct-q4_0': { ram: '6GB+', quality: 'Good', speed: 'Very Fast' },
    };
    return requirements[model] || { ram: 'Unknown', quality: 'Unknown', speed: 'Unknown' };
  }
};
