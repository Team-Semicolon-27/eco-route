import mongoose, {model, Model, models, Schema} from "mongoose";

export interface IVehicle extends Document {
    type: string;
    purchasedOn: Date;
    numberPlate: string;
    state: string;
    city: string;
}

export const vehicleSchema = new Schema<IVehicle>({
    type: { type: String, required: true },
    purchasedOn: { type: Date, required: true },
    numberPlate: { type: String, required: true },
    state: { type: String, required: true },
    city: { type: String, required: true },
})

export const Vehicle: Model<IVehicle> = models.Vehicle || model<IVehicle>("Vehicle", vehicleSchema);