import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { CgmReading, CgmReadingDocument } from './schemas/cgm-reading.schema';
import { MoodEntry, MoodEntryDocument } from './schemas/mood-entry.schema';
import { FoodLog, FoodLogDocument } from './schemas/food-log.schema';
import { ActivityLog, ActivityLogDocument } from './schemas/activity-log.schema';

@Injectable()
export class HealthDataService {
  constructor(
    @InjectModel(CgmReading.name) private cgmReadingModel: Model<CgmReadingDocument>,
    @InjectModel(MoodEntry.name) private moodEntryModel: Model<MoodEntryDocument>,
    @InjectModel(FoodLog.name) private foodLogModel: Model<FoodLogDocument>,
    @InjectModel(ActivityLog.name) private activityLogModel: Model<ActivityLogDocument>,
  ) {}

  // CGM Methods
  async saveCgmReadings(userId: string, readings: any[]): Promise<any[]> {
    const docs = readings.map((r) => ({
      ...r,
      userId,
      timestamp: r.timestamp ? new Date(r.timestamp) : new Date(),
    }));
    return this.cgmReadingModel.insertMany(docs);
  }

  async getCgmSummary(userId: string, period?: string, from?: Date, to?: Date): Promise<any> {
    const query: any = { userId };
    if (from || to) {
      query.timestamp = {};
      if (from) query.timestamp.$gte = from;
      if (to) query.timestamp.$lte = to;
    } else if (period) {
      const hours = period === '24h' ? 24 : period === '7d' ? 168 : period === '30d' ? 720 : 24;
      query.timestamp = { $gte: new Date(Date.now() - hours * 60 * 60 * 1000) };
    }

    const readings = await this.cgmReadingModel.find(query).sort({ timestamp: 1 });

    if (readings.length === 0) {
      return {
        avgGlucose: 0,
        minGlucose: 0,
        maxGlucose: 0,
        timeInRangePct: 0,
        spikeCount: 0,
        hypoCount: 0,
        readingCount: 0,
      };
    }

    const glucoseValues = readings.map((r) => r.glucoseMgDl);
    const avgGlucose = glucoseValues.reduce((a, b) => a + b, 0) / glucoseValues.length;
    const minGlucose = Math.min(...glucoseValues);
    const maxGlucose = Math.max(...glucoseValues);
    const timeInRange = glucoseValues.filter((g) => g >= 70 && g <= 140).length;
    const timeInRangePct = (timeInRange / glucoseValues.length) * 100;
    const spikeCount = glucoseValues.filter((g) => g > 180).length;
    const hypoCount = glucoseValues.filter((g) => g < 70).length;

    return {
      avgGlucose: Math.round(avgGlucose * 10) / 10,
      minGlucose,
      maxGlucose,
      timeInRangePct: Math.round(timeInRangePct * 10) / 10,
      spikeCount,
      hypoCount,
      readingCount: readings.length,
    };
  }

  async getCgmHistory(
    userId: string,
    from?: Date,
    to?: Date,
    limit?: number,
  ): Promise<CgmReadingDocument[]> {
    const query: any = { userId };
    if (from || to) {
      query.timestamp = {};
      if (from) query.timestamp.$gte = from;
      if (to) query.timestamp.$lte = to;
    }
    return this.cgmReadingModel
      .find(query)
      .sort({ timestamp: -1 })
      .limit(limit || 100);
  }

  // Mood Methods
  async saveMoodEntry(userId: string, moodData: any): Promise<MoodEntryDocument> {
    return this.moodEntryModel.create({
      ...moodData,
      userId,
      timestamp: moodData.timestamp ? new Date(moodData.timestamp) : new Date(),
    });
  }

  async getMoodHistory(
    userId: string,
    from?: Date,
    to?: Date,
    limit?: number,
  ): Promise<MoodEntryDocument[]> {
    const query: any = { userId };
    if (from || to) {
      query.timestamp = {};
      if (from) query.timestamp.$gte = from;
      if (to) query.timestamp.$lte = to;
    }
    return this.moodEntryModel
      .find(query)
      .sort({ timestamp: -1 })
      .limit(limit || 50);
  }

  // Food Methods
  async saveFoodLog(userId: string, foodData: any): Promise<FoodLogDocument> {
    return this.foodLogModel.create({
      ...foodData,
      userId,
      timestamp: foodData.timestamp ? new Date(foodData.timestamp) : new Date(),
    });
  }

  async getFoodHistory(
    userId: string,
    from?: Date,
    to?: Date,
    limit?: number,
  ): Promise<FoodLogDocument[]> {
    const query: any = { userId };
    if (from || to) {
      query.timestamp = {};
      if (from) query.timestamp.$gte = from;
      if (to) query.timestamp.$lte = to;
    }
    return this.foodLogModel
      .find(query)
      .sort({ timestamp: -1 })
      .limit(limit || 50);
  }
}
