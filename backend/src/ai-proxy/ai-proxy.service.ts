import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';
import { catchError, timeout } from 'rxjs/operators';

@Injectable()
export class AiProxyService {
  private aiServiceUrl: string;

  constructor(
    private httpService: HttpService,
    private configService: ConfigService,
  ) {
    this.aiServiceUrl = this.configService.get<string>('AI_SERVICE_URL') || 'http://localhost:8000';
  }

  async chat(userId: string, message: string, token: string): Promise<any> {
    try {
      const response = await firstValueFrom(
        this.httpService
          .post(
            `${this.aiServiceUrl}/api/v1/chat`,
            { message, user_id: userId },
            { headers: { Authorization: `Bearer ${token}` } },
          )
          .pipe(timeout(30000)),
      );
      return response.data;
    } catch (error) {
      throw new HttpException(
        'AI service temporarily unavailable',
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  async processCgmReadings(userId: string, readings: any[], token: string): Promise<any> {
    try {
      const response = await firstValueFrom(
        this.httpService
          .post(
            `${this.aiServiceUrl}/api/v1/cgm/readings`,
            { user_id: userId, readings },
            { headers: { Authorization: `Bearer ${token}` } },
          )
          .pipe(timeout(30000)),
      );
      return response.data;
    } catch (error) {
      throw new HttpException(
        'AI service temporarily unavailable',
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  async logMood(userId: string, moodData: any, token: string): Promise<any> {
    try {
      const response = await firstValueFrom(
        this.httpService
          .post(
            `${this.aiServiceUrl}/api/v1/mood`,
            { ...moodData, user_id: userId },
            { headers: { Authorization: `Bearer ${token}` } },
          )
          .pipe(timeout(30000)),
      );
      return response.data;
    } catch (error) {
      throw new HttpException(
        'AI service temporarily unavailable',
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  async logFood(userId: string, foodData: any, token: string): Promise<any> {
    try {
      const response = await firstValueFrom(
        this.httpService
          .post(
            `${this.aiServiceUrl}/api/v1/food/log`,
            { ...foodData, user_id: userId },
            { headers: { Authorization: `Bearer ${token}` } },
          )
          .pipe(timeout(30000)),
      );
      return response.data;
    } catch (error) {
      throw new HttpException(
        'AI service temporarily unavailable',
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  async getChatHistory(userId: string, token: string): Promise<any> {
    try {
      const response = await firstValueFrom(
        this.httpService
          .get(`${this.aiServiceUrl}/api/v1/history`, {
            params: { user_id: userId },
            headers: { Authorization: `Bearer ${token}` },
          })
          .pipe(timeout(30000)),
      );
      return response.data;
    } catch (error) {
      throw new HttpException(
        'AI service temporarily unavailable',
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }
}
